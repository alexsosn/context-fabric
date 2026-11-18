"""The text formats the browser offers (`formats.py`)."""

from dataclasses import replace
from pathlib import Path

from cfabric_browser import formats
from cfabric_browser.corpora import CorpusService, CorpusSpec
from cfabric_browser.display import RenderOptions

DATA = Path(__file__).resolve().parent / "fixtures" / "display_corpus" / "tf" / "1.0"


def config(text_formats, default="text-orig-full"):
    service = CorpusService([CorpusSpec(name="alpha", path=DATA)])
    return replace(
        service.config("alpha"), text_formats=text_formats, default_format=default
    )


BHSA = (
    "lex-default",
    "lex-orig-full",
    "text-orig-full",
    "text-orig-full-ketiv",
    "text-orig-plain",
    "text-trans-full",
    "text-trans-plain",
)


def test_bhsa_keeps_vocalized_and_unvocalized():
    c = config(BHSA)
    assert formats.shown(c) == ["text-orig-full", "text-orig-plain", "text-trans-full"]
    assert formats.default(c) == "text-orig-full"


def test_other_corpora_keep_the_full_text():
    c = config(("lex-default", "text-orig-extra", "text-orig-full", "text-trans-full"))
    assert formats.shown(c) == ["text-orig-full", "text-trans-full"]


def test_chosen():
    c = config(BHSA)
    assert formats.chosen(c, "text-orig-plain") == "text-orig-plain"
    assert formats.chosen(c, "text-trans-full") == "text-trans-full"
    assert formats.chosen(c, None) is None
    # a format of another corpus
    assert formats.chosen(config(("text-orig-full",)), "text-orig-plain") is None


def test_options():
    c = config(BHSA)
    opts = RenderOptions(fmt="text-orig-full-ketiv", full=True)
    assert formats.options(c, opts) == replace(opts, fmt=None)
    kept = RenderOptions(fmt="text-orig-plain")
    assert formats.options(c, kept) is kept


def test_display():
    d = formats.display(config(BHSA))
    assert d["textFormats"] == ["text-orig-full", "text-orig-plain", "text-trans-full"]
    assert d["defaultFormat"] == "text-orig-full"

def test_cuneiform_transliteration_is_selectable_and_respects_source_default():
    # The source-attested ORACC app exposes cuneiform and word transliteration.
    c = config(("text-orig-full", "text-trans-full"), default="text-trans-full")
    assert formats.shown(c) == ["text-orig-full", "text-trans-full"]
    assert formats.default(c) == "text-trans-full"
    assert formats.chosen(c, "text-trans-full") == "text-trans-full"
    selected = RenderOptions(fmt="text-trans-full")
    assert formats.options(c, selected) is selected
    assert formats.display(c)["defaultFormat"] == "text-trans-full"


def test_no_fabricated_transliteration_when_corpus_lacks_it():
    c = config(("text-orig-full",))
    assert formats.shown(c) == ["text-orig-full"]
    assert formats.chosen(c, "text-trans-full") is None


def test_transliteration_only_corpus_does_not_fabricate_original_script():
    c = config(("text-trans-full",), default="text-trans-full")
    assert formats.shown(c) == ["text-trans-full"]
    assert formats.default(c) == "text-trans-full"
