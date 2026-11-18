import { describe, expect, it } from "vitest";
import { formatOptions, formatNote, shownFormat } from "./formats";

describe("formatOptions", () => {
  it("names the BHSA's two formats", () => {
    expect(formatOptions(["text-orig-full", "text-orig-plain"], "hbo")).toEqual([
      { value: "text-orig-full", label: "Vocalized" },
      { value: "text-orig-plain", label: "Unvocalized" },
    ]);
  });

  it("offers nothing when there is no choice", () => {
    expect(formatOptions(["text-orig-full"])).toEqual([]);
    expect(formatOptions([])).toEqual([]);
  });
});

describe("shownFormat", () => {
  const formats = ["text-orig-full", "text-orig-plain"];

  it("keeps a choice the corpus offers", () => {
    expect(shownFormat("text-orig-plain", formats, "text-orig-full")).toBe("text-orig-plain");
  });

  it("gives the default for a format no longer offered, or none chosen", () => {
    expect(shownFormat("text-trans-full", formats, "text-orig-full")).toBe("text-orig-full");
    expect(shownFormat(null, formats, "text-orig-full")).toBe("text-orig-full");
  });
});

describe("source-attested script choices", () => {
  const formats = ["text-orig-full", "text-trans-full"];
  it("labels cuneiform and transliteration for Akkadian and Sumerian", () => {
    for (const language of ["akk", "sux"]) {
      expect(formatOptions(formats, language)).toEqual([
        { value: "text-orig-full", label: "Cuneiform" },
        { value: "text-trans-full", label: "Transliteration" },
      ]);
      expect(formatNote(language)).toContain("transliteration");
    }
  });

  it("retains the attested default transliteration and explicit cuneiform selection", () => {
    expect(shownFormat(null, formats, "text-trans-full")).toBe("text-trans-full");
    expect(shownFormat("text-orig-full", formats, "text-trans-full")).toBe("text-orig-full");
  });

  it("does not mislabel non-Hebrew script choices with ketiv/qere terminology", () => {
    expect(formatOptions(["text-orig-full", "text-trans-full"], "grc")[0].label)
      .toBe("Original script");
    expect(formatNote("grc")).not.toContain("ketiv");
  });

  it("labels Hebrew vocalization explicitly, without losing added transliteration", () => {
    expect(formatOptions(["text-orig-full", "text-orig-plain", "text-trans-full"], "hbo")).toEqual([
      { value: "text-orig-full", label: "Vocalized" },
      { value: "text-orig-plain", label: "Unvocalized" },
      { value: "text-trans-full", label: "Transliteration" },
    ]);
    expect(formatNote("hbo")).toContain("ketiv");
  });

  it("does not offer a choice when the source supplies only one text format", () => {
    expect(formatOptions(["text-trans-full"], "akk")).toEqual([]);
  });
});
