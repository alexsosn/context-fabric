// Names and help are specific to a corpus writing system, never guessed from
// a Text-Fabric format identifier alone.
export function formatNote(languageCode: string | null = null): string {
  if (languageCode === "hbo" || languageCode === "heb")
    return "Vocalized: with vowels and accents, and the qere where it differs from the ketiv. Unvocalized: the consonants of the ketiv.";
  if (languageCode === "akk" || languageCode === "sux")
    return "Cuneiform shows source-provided signs. Transliteration follows the source reading; missing signs are not reconstructed.";
  return "Available scripts and transliterations are provided by this corpus.";
}

export function formatOptions(
  formats: string[],
  languageCode: string | null = null,
): { value: string; label: string }[] {
  if (formats.length < 2) return [];
  const hebrew = languageCode === "hbo" || languageCode === "heb";
  const cuneiform = languageCode === "akk" || languageCode === "sux";
  const labels: Record<string, string> = {
    "text-orig-full": hebrew ? "Vocalized" : cuneiform ? "Cuneiform" : "Original script",
    "text-orig-plain": hebrew ? "Unvocalized" : "Original script (plain)",
    "text-trans-full": "Transliteration",
  };
  return formats.map((f) => ({ value: f, label: labels[f] ?? f }));
}

export function shownFormat(chosen: string | null, formats: string[], defaultFormat: string | null): string | null {
  return chosen !== null && formats.includes(chosen) ? chosen : defaultFormat;
}
