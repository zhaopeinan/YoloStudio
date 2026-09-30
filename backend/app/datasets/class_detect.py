from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Image


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CLASS_COLORS = [
    "#e45756",
    "#2f80ed",
    "#27ae60",
    "#f2994a",
    "#9b51e0",
    "#1f6f78",
    "#d97706",
    "#dc2626",
    "#00a3a3",
    "#6366f1",
]

# Domain pattern: fire_truck_h20_a045.jpg -> fire_truck
HA_SUFFIX_RE = re.compile(r"^(?P<prefix>.+?)_h\d+_a\d+$", re.IGNORECASE)
# Fallback: strip trailing numeric / pose tokens like _00012, _z12m, _frame3
TRAILING_TOKEN_RE = re.compile(
    r"^(?P<prefix>.+?)(?:_(?:h\d+|a\d+|z\d+m?|frame\d*|\d+))+$",
    re.IGNORECASE,
)


def extract_class_prefix(filename: str) -> str | None:
    stem_name = Path(filename).name
    if stem_name.startswith("._") or stem_name.lower() in {".ds_store", "thumbs.db", "desktop.ini"}:
        return None
    stem = Path(stem_name).stem.strip()
    if not stem or stem.startswith("._"):
        return None
    match = HA_SUFFIX_RE.match(stem)
    if match:
        prefix = match.group("prefix").strip("_- ")
        if not prefix or prefix.startswith("._"):
            return None
        return prefix
    match = TRAILING_TOKEN_RE.match(stem)
    if match:
        prefix = match.group("prefix").strip("_- ")
        # Avoid treating whole stem as class when almost nothing was stripped.
        if prefix and prefix != stem and len(prefix) >= 2 and not prefix.startswith("._"):
            return prefix
    return None


def detect_classes_from_filenames(
    filenames: list[str],
    *,
    min_count: int = 2,
) -> tuple[list[dict], str]:
    """Return detected classes sorted by count desc, then name.

    Prefers the h/a pose naming scheme used by the drone dataset; falls back to
    stripping trailing numeric tokens when that covers enough images.
    """
    samples: dict[str, list[str]] = defaultdict(list)

    usable_filenames = [
        name
        for name in filenames
        if not Path(name).name.startswith("._")
        and Path(name).name.lower() not in {".ds_store", "thumbs.db", "desktop.ini"}
    ]

    ha_counts: Counter[str] = Counter()
    for name in usable_filenames:
        stem_name = Path(name).name
        match = HA_SUFFIX_RE.match(Path(stem_name).stem)
        if not match:
            continue
        prefix = match.group("prefix").strip("_- ")
        if not prefix or prefix.startswith("._"):
            continue
        ha_counts[prefix] += 1
        if len(samples[prefix]) < 3:
            samples[prefix].append(stem_name)

    total = len(usable_filenames)
    ha_covered = sum(ha_counts.values())
    if ha_counts and ha_covered >= max(min_count, int(total * 0.3)):
        return _to_items(ha_counts, samples, min_count), "filename_prefix_h_a"

    fallback_counts: Counter[str] = Counter()
    samples = defaultdict(list)
    for name in usable_filenames:
        stem_name = Path(name).name
        prefix = extract_class_prefix(stem_name)
        if not prefix:
            continue
        # Prefer HA extraction inside extract when present.
        fallback_counts[prefix] += 1
        if len(samples[prefix]) < 3:
            samples[prefix].append(stem_name)

    if fallback_counts:
        return _to_items(fallback_counts, samples, min_count), "filename_prefix"

    return [], "none"


def _to_items(
    counts: Counter[str],
    samples: dict[str, list[str]],
    min_count: int,
) -> list[dict]:
    items: list[dict] = []
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    color_index = 0
    for name, count in ranked:
        if count < min_count:
            continue
        items.append(
            {
                "name": name,
                "image_count": count,
                "color": DEFAULT_CLASS_COLORS[color_index % len(DEFAULT_CLASS_COLORS)],
                "sample_filenames": samples.get(name, [])[:3],
            }
        )
        color_index += 1
    return items


def detect_classes_for_dataset(db: Session, dataset_id: int) -> dict:
    images = list(
        db.scalars(select(Image).where(Image.dataset_id == dataset_id).order_by(Image.id)).all()
    )
    filenames = [Path(image.relative_path).name for image in images]
    items, method = detect_classes_from_filenames(filenames)
    return {
        "dataset_id": dataset_id,
        "total_images": len(filenames),
        "method": method,
        "items": items,
    }
