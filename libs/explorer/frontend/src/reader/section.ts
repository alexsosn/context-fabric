/** A selected section is identified by its actual TF reference.
 *
 * Numeric Bible verses retain the compact `?v=3` URL. For two-level
 * inscriptions or qualified ORACC source IDs, the exact line reference is
 * kept in `?line=` and the parent is supplied by the API's chapter container.
 * No textual ID is parsed or normalized to invent a verse number.
 */
export function selectionForRef(ref: string, chapter: string): {
  chapter: string;
  verse: number | null;
  selectedRef: string | null;
} {
  const prefix = chapter + ":";
  if (ref.startsWith(prefix)) {
    const last = ref.slice(prefix.length);
    if (/^\d+$/.test(last)) {
      const verse = Number(last);
      if (Number.isSafeInteger(verse)) return { chapter, verse, selectedRef: null };
    }
  }
  return { chapter, verse: null, selectedRef: ref };
}

/** Show a source-attested terminal section label, keeping the complete ref
 * for selection, citation and links. Never shorten an unrelated identifier.
 */
export function sectionLabel(ref: string, parent: string): string {
  for (const sep of [":", " "]) {
    if (ref.startsWith(parent + sep)) return ref.slice(parent.length + sep.length);
  }
  return ref;
}

/** A deep-link query parameter is not proof that a line exists in the source. */
export function verifiedSectionRef(
  ref: string | null,
  trees: readonly { heading: string }[] | undefined,
): string | null {
  return ref && trees?.some((tree) => tree.heading === ref) ? ref : null;
}
