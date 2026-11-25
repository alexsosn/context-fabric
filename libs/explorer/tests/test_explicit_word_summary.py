"""RED contracts for explicit, source-preserving lexical display roles.

Witness values transcribed from pinned ORACC-TF source blobs:
ETCSRI Q000935.1.1 (a2eef2a0337d276191fbda85a4269948fb00f9c3);
NINMED P285136.1.1 (a26ee7550a1f6d62187ef84b7d84ab1bfd877b2b).
These unit tests use those values, not a loaded TF corpus. Real TF route
and interlinear integration tests are still required before merging.
"""

from types import SimpleNamespace

from cfabric_browser.interlinear import word_lines
from cfabric_browser.summary import summary


ROLES = {
    "transliteration": "form",
    "citation form": "cf",
    "guide word": "gw",
    "sense": "sense",
    "part of speech": "pos",
    "language": "lang",
    "morphology": "morph",
    "morphology 2": "morph2",
    "source annotation flag": "bad",
}


def corpus(**features):
    return SimpleNamespace(
        F=SimpleNamespace(
            **{
                name: SimpleNamespace(v=lambda n, values=values: values.get(n))
                for name, values in features.items()
            }
        )
    )


def test_etcsri_source_roles_preserve_guide_word_and_sense_separately():
    api = corpus(
        form={1: "ur-{d}namma"},
        cf={1: "Urnammak"},
        gw={1: "1"},
        sense={1: "1st king of the 3rd dynasty of Urim"},
        pos={1: "RN"},
        lang={1: "sux"},
        morph={1: "N1=Urnammak"},
        morph2={1: "N1=NAME"},
    )
    items = summary(api, 1, roles=ROLES)
    assert items == [
        {"label": label, "value": getattr(api.F, feature).v(1), "feature": feature}
        for label, feature in ROLES.items()
        if feature != "bad"
    ]
    assert word_lines(items) == {
        "translit": "ur-{d}namma",
        "lexeme": "Urnammak",
        "gloss": "1st king of the 3rd dynasty of Urim",
        "pos": "RN",
        "morph": "N1=Urnammak / N1=NAME",
        "language": "sux",
    }


def test_ninmed_source_roles_keep_diacritics_dialect_and_bad_flag():
    api = corpus(
        form={1: "DU₃.DU₃.BI"},
        cf={1: "DÙ.DÙ.BI"},
        gw={1: "ritual (instructions)"},
        pos={1: "N"},
        lang={1: "akk-x-stdbab"},
        bad={1: "yes"},
        # An unrelated ETCBC-shaped feature must not trigger Hebrew cleaning,
        # normalization or an external glossary lookup in an explicit profile.
        lex={1: "TEST/"},
    )

    def unexpected_lookup(*_):
        raise AssertionError("Explicit roles must not borrow ETCBC glosses")

    items = summary(api, 1, gloss_for=unexpected_lookup, roles=ROLES)
    assert items == [
        {"label": label, "value": getattr(api.F, feature).v(1), "feature": feature}
        for label, feature in ROLES.items()
        if feature not in {"sense", "morph", "morph2"}
    ]
    assert word_lines(items) == {
        "translit": "DU₃.DU₃.BI",
        "lexeme": "DÙ.DÙ.BI",
        "gloss": "ritual (instructions)",
        "pos": "N",
        "language": "akk-x-stdbab",
    }


def test_explicit_roles_do_not_guess_missing_values_or_hide_source_zero():
    api = corpus(cf={1: ""}, gw={1: "0"}, morph={1: "?"}, lex={1: "ABC/"})
    assert summary(api, 1, {"?"}, roles=ROLES) == [
        {"label": "guide word", "value": "0", "feature": "gw"}
    ]


def test_explicit_empty_profile_does_not_fall_back_to_hebrew_roles():
    api = corpus(gloss={1: "unrelated"}, lex={1: "ABC/"})
    assert summary(api, 1, roles={}) == []
