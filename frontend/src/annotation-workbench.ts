type SelectableImage = {
  id: number;
  annotation_count: number;
  annotation_status?: "unreviewed" | "annotated" | "negative";
};

function isUnreviewed(image: SelectableImage): boolean {
  return image.annotation_status
    ? image.annotation_status === "unreviewed"
    : image.annotation_count === 0;
}

export type AdvancedImageFilters = {
  platform: string;
  class_id: string;
  edge_tag: string;
  failure_type: string;
  altitude_min: string;
  altitude_max: string;
};

export function selectInitialAnnotationImageId(
  images: SelectableImage[],
  currentId: number | null,
): number | null {
  if (images.some((image) => image.id === currentId)) return currentId;
  return images.find(isUnreviewed)?.id ?? images[0]?.id ?? null;
}

export function findNextAnnotationImageId(
  images: SelectableImage[],
  currentId: number,
): number | null {
  const currentIndex = images.findIndex((image) => image.id === currentId);
  if (currentIndex < 0) return selectInitialAnnotationImageId(images, null);
  const remaining = images.slice(currentIndex + 1);
  return remaining.find(isUnreviewed)?.id ?? remaining[0]?.id ?? null;
}

export function countAdvancedImageFilters(filters: AdvancedImageFilters): number {
  return Object.entries(filters).filter(([key, value]) =>
    key === "failure_type" ? value !== "all" : value.trim() !== "",
  ).length;
}
