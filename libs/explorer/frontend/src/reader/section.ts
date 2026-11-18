import type { Panel } from "./url";

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

/** A two-level corpus has document → line, so start at the document.
 * Three-level corpora (e.g. Bible book/chapter/verse or ORACC text/face/line)
 * start at the first containing section, never at an isolated leaf.
 */
export function initialReadingRef(
  document: string,
  firstChild: string | null,
  sectionLevelCount: number,
): string {
  return sectionLevelCount === 2 ? document : (firstChild ?? document);
}

/** Avoid opening a Bible-only parallel verse panel for source-qualified lines. */
export function panelAfterSelection(
  current: Panel | null,
  verse: number | null,
  selectedRef: string | null,
): Panel | null {
  return current ?? (selectedRef !== null ? "words" : (verse !== null ? "parallels" : null));
}

/** Follow an explicit TF section path for a Reader deep link.
 * Do not infer a line's parent by splitting its source-qualified string.
 */
export function sectionNavigation(
  ref: string,
  path: readonly { type: string; ref: string }[],
  leafType: string,
): { chapter: string; selectedRef: string | null } {
  const leaf = path[path.length - 1];
  const parent = path[path.length - 2];
  if (leaf?.ref === ref && leaf.type === leafType && parent) {
    return { chapter: parent.ref, selectedRef: ref };
  }
  return { chapter: ref, selectedRef: null };
}
