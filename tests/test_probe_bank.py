"""probe_bank.pick(): which probe .pt scores this instruction (torch-free)."""

from vla.cortex import SubtaskMachine
from vla.progress_probe import override_from_args, pick

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
