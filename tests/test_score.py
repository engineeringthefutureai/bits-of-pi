import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from score import (align, copy_metrics, extract_digits, extract_recall, extract_words, in_run,
                   planted_outcome, prefix_len)


def test_extract_recall_stops_at_first_non_digit():
    assert extract_recall("14159x26") == ("14159", "14159")
    assert extract_recall("14159\n26535...") == ("14159", "1415926535")
    assert extract_recall("") == ("", "")


def test_extract_digits_drops_integer_part_and_noise():
    assert extract_digits("3.14159", "3") == "14159"
    assert extract_digits("  3 .14 15\n9", "3") == "14159"
    assert extract_digits("14159", "3") == "14159"
    assert extract_digits("314159", "3") == "14159"
    assert extract_digits("Here: 0123", None) == "0123"


def test_extract_words():
    assert extract_words("one two Three, nine.") == ("1239", [])
    assert extract_words("three point one four", "3") == ("14", [])
    assert extract_words("Three point one four", None) == ("314", ["point"])
    digits, errors = extract_words("one tow two")
    assert digits == "12" and errors == ["tow"]


def test_prefix_len():
    assert prefix_len("14159", "14159") == 5
    assert prefix_len("14159", "14169") == 3
    assert prefix_len("14159", "141") == 3


def ops_of(ref, hyp):
    return [op for op, _, _ in align(ref, hyp) if op != "="]


def test_align_identity():
    assert ops_of("0123456789", "0123456789") == []


def test_align_single_deletion_is_one_error_not_many():
    ref = "0123456789" * 30
    hyp = ref[:100] + ref[101:]
    m, _ = copy_metrics(ref, hyp)
    assert (m["edit_distance"], m["dels"], m["subs"], m["ins"]) == (1, 1, 0, 0)
    assert m["prefix"] == 100


def test_align_insertion_and_substitution():
    m, _ = copy_metrics("12345", "123945")
    assert (m["ins"], m["edit_distance"]) == (1, 1)
    m, _ = copy_metrics("12345", "12945")
    assert (m["subs"], m["edit_distance"]) == (1, 1)


def test_align_positions_cover_both_strings():
    ref, hyp = "1122334455", "112334555"
    ops = align(ref, hyp)
    assert sorted(rp for op, rp, _ in ops if op != "I") == list(range(len(ref)))
    assert sorted(hp for op, _, hp in ops if op != "D") == list(range(len(hyp)))


def test_in_run():
    ref = "12777774"
    assert in_run(ref, 3)
    assert not in_run(ref, 0)
    assert in_run(ref, 7)  # boundary: insertion just after the run counts


def test_dropped_digit_inside_run_is_flagged():
    ref = "1234" + "7" * 6 + "5678"
    hyp = "1234" + "7" * 5 + "5678"
    m, _ = copy_metrics(ref, hyp)
    assert m["dels"] == 1 and m["errors_in_run"] == 1


def test_planted_outcome():
    ref = "1415926535"  # planted at 4: true digit 9 replaced by 4
    planted = ref[:4] + "4" + ref[5:]
    for hyp, want in [(planted, "planted"), (ref, "memory"), (ref[:4] + "0" + ref[5:], "other"),
                      (ref[:4] + ref[5:], "other")]:
        _, ops = copy_metrics(planted, hyp)
        assert planted_outcome(ops, hyp, 4, "9", "4") == want, hyp
