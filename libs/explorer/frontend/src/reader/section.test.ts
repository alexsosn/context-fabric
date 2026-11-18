import { describe, expect, it } from "vitest";
import { selectionForRef, sectionLabel, verifiedSectionRef, initialReadingRef, panelAfterSelection } from "./section";
import { parseReader, serializeReader, type Reader } from "./url";

const riaoFace = "riao/ria1:Q001801 Q001801.face.1";
const riaoLine = "riao/ria1:Q001801 Q001801.face.1:Q001801.1";
const ninmedFace = "asbp/ninmed:P285136 P285136.face.1";
const ninmedLine = "asbp/ninmed:P285136 P285136.face.1:P285136.1";

describe("Reader section selection", () => {
  it("keeps a Bible verse numeric in its parent chapter", () => {
    expect(selectionForRef("Genesis 1:3", "Genesis 1")).toEqual({
      chapter: "Genesis 1", verse: 3, selectedRef: null,
    });
    expect(sectionLabel("Genesis 1:3", "Genesis 1")).toBe("3");
  });

  it("keeps real ORACC qualified 3-level line refs intact", () => {
    for (const [chapter, ref, label] of [
      [riaoFace, riaoLine, "Q001801.1"],
      [ninmedFace, ninmedLine, "P285136.1"],
    ]) {
      expect(selectionForRef(ref, chapter)).toEqual({
        chapter, verse: null, selectedRef: ref,
      });
      expect(sectionLabel(ref, chapter)).toBe(label);
    }
  });

  it("keeps a real 2-level ETCSRI line ref and its source label intact", () => {
    const chapter = "Q000935";
    const ref = "Q000935 Q000935.1";
    expect(selectionForRef(ref, chapter)).toEqual({
      chapter, verse: null, selectedRef: ref,
    });
    expect(sectionLabel(ref, chapter)).toBe("Q000935.1");
  });

  it("does not shorten an unrelated section or misread an ID as a verse", () => {
    expect(selectionForRef("riao/ria1:Q001801", "Some other text").selectedRef)
      .toBe("riao/ria1:Q001801");
    expect(sectionLabel("asbp/ninmed:P285136", "Some other text"))
      .toBe("asbp/ninmed:P285136");
  });

  it("round-trips exact selected ORACC source IDs through the Reader URL", () => {
    const initial: Reader = {
      corpus: "riao", ref: riaoFace, verse: null, selectedRef: riaoLine,
      panel: "words", mode: "verses", texts: null, diff: false,
    };
    const url = new URL(serializeReader(initial), "http://localhost");
    const parsed = parseReader(url);
    expect(parsed).toEqual(initial);
    expect(parsed.selectedRef).toBe(riaoLine);
  });

  it("keeps existing Bible verse URLs backward-compatible", () => {
    const state = parseReader({ pathname: "/read/bhsa/Genesis/1", search: "?v=3" });
    expect(state.ref).toBe("Genesis 1");
    expect(state.verse).toBe(3);
    expect(state.selectedRef).toBeUndefined();
    expect(serializeReader(state)).toBe("/read/bhsa/Genesis/1?v=3");
  });
  it("never cites an unverified line injected through a URL", () => {
    expect(verifiedSectionRef(riaoLine, [{ heading: riaoLine }])).toBe(riaoLine);
    expect(verifiedSectionRef("riao/ria1:FORGED", [{ heading: riaoLine }])).toBeNull();
    expect(verifiedSectionRef("Genesis 1:99999", [{ heading: "Genesis 1:3" }])).toBeNull();
    expect(verifiedSectionRef(riaoLine, undefined)).toBeNull();
  });
  it("opens a whole document for 2-level TF, not only its first line", () => {
    expect(initialReadingRef("Q000935", "Q000935 1", 2)).toBe("Q000935");
    expect(initialReadingRef("Genesis", "Genesis 1", 3)).toBe("Genesis 1");
    expect(initialReadingRef("riao/ria1:Q001801", riaoFace, 3)).toBe(riaoFace);
    expect(initialReadingRef("OneDocument", null, 1)).toBe("OneDocument");
  });

  it("opens source-word inspection by default for unnumbered ORACC lines", () => {
    expect(panelAfterSelection(null, null, riaoLine)).toBe("words");
    expect(panelAfterSelection(null, 3, null)).toBe("parallels");
    expect(panelAfterSelection("notes", null, riaoLine)).toBe("notes");
    expect(panelAfterSelection(null, null, null)).toBeNull();
  });

});
