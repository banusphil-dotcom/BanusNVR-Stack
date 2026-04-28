// Shared display helpers for raw COCO/Frigate object types.
//
// Frigate's detector regularly mis-labels cats as dogs (and vice versa) on
// our cameras, so for animal detections we collapse cat/dog/bird into a
// single "pet" label across the entire UI. Recognised animals always show
// their named object name regardless.

const PET_TYPES = new Set(["cat", "dog", "bird"]);

export function isPetType(t: string | null | undefined): boolean {
  return !!t && PET_TYPES.has(t.toLowerCase());
}

/** Human-friendly object label, lowercased. Returns null for empty input. */
export function prettyObjectType(t: string | null | undefined): string | null {
  if (!t) return null;
  return isPetType(t) ? "pet" : t;
}

/** Title-cased version for notification text / badges. */
export function prettyObjectTypeTitle(t: string | null | undefined): string {
  const v = prettyObjectType(t);
  if (!v) return "";
  return v.charAt(0).toUpperCase() + v.slice(1);
}
