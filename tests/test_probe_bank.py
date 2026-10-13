"""probe_bank.pick(): which probe .pt scores this instruction (torch-free)."""

import json

import pytest

from vla.cortex import SubtaskMachine
from vla.progress_probe import override_from_args, pick
from vla.progress_probe.probe_bank import load_index

PICK = ("Reach out with the right hand and pick up the cucumber inside", "/p/probe_pick.pt")
PLACE = ("Place the cucumber on the plate", "/p/probe_place.pt")
INDEX = [PICK, PLACE]


def test_exact_prompt_match():
    assert pick(INDEX, PLACE[0]).path == PLACE[1]


def test_no_match_is_none_not_a_guess():
    assert pick(INDEX, "Open the fridge.").path is None


def test_override_by_stem_wins_over_prompt():
    assert pick(INDEX, PICK[0], override="probe_place").path == PLACE[1]


def test_unknown_override_is_none_never_fallback():
    assert pick(INDEX, PICK[0], override="probe_v9").path is None


def test_override_from_args_strips_pt():
    assert override_from_args(["x=1", "probe=probe_place.pt"]) == "probe_place"
    assert override_from_args(["x=1"]) is None


def test_machine_exposes_probe_override_while_running():
    m = SubtaskMachine()
    assert m.probe_override() is None
    m.on_cmd(plan_id="p", index=0, action="pick", instruction="Pick.", cancel=False,
             now=1.0, args=("probe=probe_pick_v2",))
    assert m.probe_override() == "probe_pick_v2"
    m.on_cmd(plan_id="p", index=0, action="", instruction="", cancel=True, now=2.0)
    assert m.probe_override() is None


def test_load_index_splits_heads_and_timeouts(tmp_path):
    p = tmp_path / "probes.json"
    p.write_text(json.dumps({
        PICK[0]: "probe_pick.pt",
        "Walk up to the fridge.": 8,
        "Step back from the refrigerator.": 5.5,
    }))
    heads, timeouts = load_index(str(p))
    assert heads == {PICK[0]: str(tmp_path / "probe_pick.pt")}
    assert timeouts == {"Walk up to the fridge.": 8.0, "Step back from the refrigerator.": 5.5}


def test_load_index_rejects_other_values(tmp_path):
    p = tmp_path / "probes.json"
    p.write_text(json.dumps({"Walk.": True}))
    with pytest.raises(ValueError):
        load_index(str(p))
    p.write_text(json.dumps({"Walk.": ["probe_pick.pt"]}))
    with pytest.raises(ValueError):
        load_index(str(p))
