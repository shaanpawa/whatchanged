"""Score a run against a hand-written answer key (ground_truth.yaml).

The key lists what was really changed, written before running anything:

    changes:
      - {id: 1, type: moved,   names: [desk chair, chair], distance_m: 1.0}
      - {id: 2, type: added,   names: [suitcase, luggage, bag]}
      - {id: 3, type: removed, names: [floor lamp, lamp]}
      - {id: 5, type: moved,   names: [mug, cup], expected: miss}
    traps:
      - {id: T1, kind: hidden, names: [shoes]}

A detection matches a key entry when the change type agrees and the label shares a word with
one of the names. "Expected miss" entries are reported but not counted against recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from scipy.optimize import linear_sum_assignment


def _words(s: str) -> set[str]:
    return {w.strip(".,").lower() for w in s.split() if len(w) > 2}


def _label_ok(label: str, names: list[str]) -> bool:
    lw = _words(label)
    return any(lw & _words(n) for n in names)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for k successes out of n (honest for small n)."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(max(0.0, centre - half), 2), round(min(1.0, centre + half), 2)]


def score(changes_json: Path, key_yaml: Path) -> dict:
    found = json.loads(Path(changes_json).read_text())["changes"]
    key = yaml.safe_load(Path(key_yaml).read_text())
    gt = key.get("changes", [])
    counted = [g for g in gt if g.get("expected") != "miss"]
    cost = [[0.0 if (d["type"] == g["type"] and _label_ok(d["label"], g["names"])) else
             0.5 if d["type"] == g["type"] else 1.0 for d in found] for g in counted]
    matches = []
    if counted and found:
        rows, cols = linear_sum_assignment(cost)
        matches = [(r, c, cost[r][c]) for r, c in zip(rows, cols) if cost[r][c] < 1.0]
    rows_matched = {r for r, _, _ in matches}
    cols_matched = {c for _, c, _ in matches}
    table = []
    for i, g in enumerate(counted):
        m = next(((c, k) for r, c, k in matches if r == i), None)
        entry = {"id": g["id"], "truth": f"{g['type']} {g['names'][0]}", "result": "missed"}
        if m:
            d = found[m[0]]
            entry.update(result="found" if m[1] == 0 else "type only", detected=f"{d['type']} {d['label']}", change=d["id"])
            if g["type"] == "moved" and "distance_m" in g and "move" in d:
                entry["distance_error_m"] = round(abs(d["move"]["distance_m"] - g["distance_m"]), 2)
        table.append(entry)
    for g in gt:
        if g.get("expected") == "miss":
            hit = any(d["type"] == g["type"] and _label_ok(d["label"], g["names"]) for d in found)
            table.append({"id": g["id"], "truth": f"{g['type']} {g['names'][0]} (expected miss)", "result": "found" if hit else "missed as expected"})
    false = [d for j, d in enumerate(found) if j not in cols_matched]
    # A trap is failed if something was reported as removed with one of the trap's names.
    traps = []
    for t in key.get("traps", []):
        bad = [d["id"] for d in found if d["type"] == "removed" and _label_ok(d["label"], t.get("names", []))]
        traps.append({"id": t["id"], "kind": t["kind"], "passed": not bad, "wrongly_reported": bad})
    n_found = sum(e["result"] == "found" for e in table if "expected miss" not in e["truth"])
    n_true = len([m for m in matches if m[2] == 0])
    out = {
        "recall_95ci": wilson(n_found, len(counted)), "precision_95ci": wilson(n_true, len(found)),
        "recall": round(n_found / max(len(counted), 1), 2),
        "precision": round(len([m for m in matches if m[2] == 0]) / max(len(found), 1), 2),
        "found": n_found, "of": len(counted), "type_only": sum(e["result"] == "type only" for e in table),
        "false_alarms": len(false), "false_alarm_list": [f"{d['type']} {d['label']} ({d['id']})" for d in false],
        "traps": traps, "table": table,
    }
    return out


def markdown(s: dict, title: str) -> str:
    lines = [f"### {title}", "",
             f"Found **{s['found']} of {s['of']}** real changes (95% interval {s['recall_95ci'][0]:.0%}-{s['recall_95ci'][1]:.0%}) · "
             f"precision {s['precision']:.0%} · "
             f"false alarms **{s['false_alarms']}** · traps passed {sum(t['passed'] for t in s['traps'])}/{len(s['traps'])}", "",
             "| # | What really changed | Result | Detected | Move error |", "|---|---|---|---|---|"]
    for e in s["table"]:
        err = f"{e['distance_error_m'] * 100:.0f} cm" if "distance_error_m" in e else ""
        lines.append(f"| {e['id']} | {e['truth']} | {e['result']} | {e.get('detected', '')} | {err} |")
    if s["false_alarm_list"]:
        lines += ["", "False alarms: " + ", ".join(s["false_alarm_list"])]
    return "\n".join(lines) + "\n"
