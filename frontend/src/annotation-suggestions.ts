import type { ProjectClass } from "./api";

export type FilenameSuggestion = {
  selected: ProjectClass | null;
  candidates: ProjectClass[];
  token: string | null;
};

const explicitAliases: Record<string, string[]> = {
  fire_truck: ["firet", "firetruck"],
  person_white: ["personwhite", "person"],
  prius_hybrid: ["priushybrid", "prius"],
  car_lexus: ["carlexus", "lexus", "lx"],
  prius_hybrid_camo: ["priushybridcamo", "priuscamo", "prius"],
  suv_camo: ["suvcamo", "suv", "suuv"],
  car_opel: ["caropel", "opel"],
  person_red: ["personred", "redperson", "person"],
};

const HA_SUFFIX_RE = /^(?<prefix>.+?)_h\d+_a\d+$/i;
const TRAILING_TOKEN_RE =
  /^(?<prefix>.+?)(?:_(?:h\d+|a\d+|z\d+m?|frame\d*|\d+))+$/i;

function normalize(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function basenameWithoutExtension(path: string): string {
  const basename = path.split(/[\\/]/).pop() ?? path;
  return basename.replace(/\.[^.]+$/, "");
}

function extractFilenameClassPrefix(stem: string): string | null {
  const haMatch = HA_SUFFIX_RE.exec(stem);
  if (haMatch?.groups?.prefix) {
    const prefix = haMatch.groups.prefix.trim().replace(/^[_-]+|[_-]+$/g, "");
    return prefix || null;
  }
  const trailingMatch = TRAILING_TOKEN_RE.exec(stem);
  if (trailingMatch?.groups?.prefix) {
    const prefix = trailingMatch.groups.prefix.trim().replace(/^[_-]+|[_-]+$/g, "");
    if (prefix && prefix !== stem && prefix.length >= 2) {
      return prefix;
    }
  }
  return null;
}

type AliasMatch = {
  classItem: ProjectClass;
  classKey: string;
  matchedAlias: string;
  score: number;
};

function dropParentClassMatches(matches: AliasMatch[]): AliasMatch[] {
  return matches.filter((candidate) => {
    return !matches.some((other) => {
      if (other.classItem.id === candidate.classItem.id) {
        return false;
      }
      if (!other.classKey.startsWith(candidate.classKey)) {
        return false;
      }
      // Only drop the shorter parent when the child matched a more specific token.
      return (
        other.matchedAlias.length > candidate.matchedAlias.length &&
        other.matchedAlias.startsWith(candidate.matchedAlias)
      );
    });
  });
}

function pickUnique(matches: AliasMatch[]): FilenameSuggestion {
  if (matches.length === 1) {
    return {
      selected: matches[0].classItem,
      candidates: [matches[0].classItem],
      token: matches[0].matchedAlias,
    };
  }
  return {
    selected: null,
    candidates: matches.map((match) => match.classItem),
    token: matches[0]?.matchedAlias ?? null,
  };
}

export function suggestClassFromFilename(
  filename: string,
  classes: ProjectClass[],
): FilenameSuggestion {
  const stem = basenameWithoutExtension(filename);
  const normalizedFilename = normalize(stem);
  if (!normalizedFilename || classes.length === 0) {
    return { selected: null, candidates: [], token: null };
  }

  const activeClasses = classes.filter((classItem) => classItem.active !== false);
  const classPool = activeClasses.length > 0 ? activeClasses : classes;

  // Prefer domain naming: prius_hybrid_camo_h10_a036.jpg -> prius_hybrid_camo
  const extractedPrefix = extractFilenameClassPrefix(stem);
  if (extractedPrefix) {
    const exact = classPool.filter(
      (classItem) => classItem.name.toLowerCase() === extractedPrefix.toLowerCase(),
    );
    if (exact.length === 1) {
      return { selected: exact[0], candidates: exact, token: extractedPrefix };
    }

    const normalizedPrefix = normalize(extractedPrefix);
    const normalizedExact = classPool.filter(
      (classItem) => normalize(classItem.name) === normalizedPrefix,
    );
    if (normalizedExact.length === 1) {
      return {
        selected: normalizedExact[0],
        candidates: normalizedExact,
        token: extractedPrefix,
      };
    }
  }

  const matches = classPool.flatMap((classItem) => {
    const classKey = normalize(classItem.name);
    const aliases = new Set(
      [classKey, ...(explicitAliases[classItem.name] ?? [])].map(normalize).filter(Boolean),
    );
    const matchedAlias = Array.from(aliases)
      .filter((alias) => normalizedFilename.includes(alias))
      .sort((left, right) => right.length - left.length)[0];
    if (!matchedAlias) {
      return [];
    }
    const score = matchedAlias === classKey ? 3 : 2;
    return [{ classItem, classKey, matchedAlias, score }];
  });

  if (matches.length === 0) {
    return { selected: null, candidates: [], token: null };
  }

  const refined = dropParentClassMatches(matches);
  const bestScore = Math.max(...refined.map((match) => match.score));
  const bestMatches = refined.filter((match) => match.score === bestScore);
  return pickUnique(bestMatches);
}
