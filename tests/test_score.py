"""The scorecard: closest names win, one end of a move is not a false alarm."""

import json

from whatchanged.score import markdown, score


def test_closest_name_wins_and_one_end_of_a_move_is_not_a_false_alarm(tmp_path):
    found = [{"id": "c01", "type": "removed", "label": "office chair"},
             {"id": "c02", "type": "moved", "label": "wooden chair"},
             {"id": "c03", "type": "added", "label": "lamp"}]
    (tmp_path / "changes.json").write_text(json.dumps({"changes": found}))
    (tmp_path / "key.yaml").write_text(
        "changes:\n"
        "  - {id: 1, type: moved, names: [office chair, desk chair]}\n"
        "  - {id: 2, type: moved, names: [wooden chair, chair]}\n"
        "  - {id: 3, type: removed, names: [speaker]}\n")
    s = score(tmp_path / "changes.json", tmp_path / "key.yaml")
    rows = {e["id"]: e for e in s["table"]}
    assert rows[2]["result"] == "found" and rows[2]["change"] == "c02"       # not given to the office chair
    assert rows[1]["result"] == "one end of the move" and rows[1]["change"] == "c01"
    assert rows[3]["result"] == "missed"
    assert (s["found"], s["one_end"], s["false_alarms"]) == (1, 1, 1) and s["false_alarm_list"] == ["added lamp (c03)"]
    assert "one end of a move for 1 more" in markdown(s, "t")
