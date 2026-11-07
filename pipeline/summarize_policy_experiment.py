"""Collect the closed-loop results of policy_experiment.sh into one table (with 95% Wilson intervals).

Example:
    python scripts/ur_cube/summarize_policy_experiment.py   # -> outputs/ur_cube/eval/summary.json + .md
"""

import json
import math
from pathlib import Path

NAMES = ["single", "r2r2r_25", "r2r2r_100", "r2r2r_400", "edit_repair_400", "edit_reject_400"]
LABELS = {
    "single": "1 demo, replayed",
    "r2r2r_25": "R2R2R, 25 rollouts",
    "r2r2r_100": "R2R2R, 100 rollouts",
    "r2r2r_400": "R2R2R, 400 rollouts",
    "edit_repair_400": "R2R2R + bottle scenes, repaired (400)",
    "edit_reject_400": "R2R2R + bottle scenes, reject-only (400)",
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main(eval_dir: Path = Path("outputs/ur_cube/eval"), policies: Path = Path("outputs/ur_cube/policies")) -> None:
    rows = []
    for name in NAMES:
        tl = policies / name / "train_log.json"
        row = dict(name=name, label=LABELS[name], rollouts=json.loads(tl.read_text())["rollouts"] if tl.exists() else None)
        for s in ("clean", "blocked"):
            f = eval_dir / f"{name}_{s}" / "results.jsonl"
            if not f.exists():
                continue
            R = [json.loads(l) for l in f.read_text().splitlines()]
            k, n = sum(r["success"] for r in R), len(R)
            lo, hi = wilson(k, n)
            row[s] = dict(success=k, n=n, rate=round(k / n, 3), ci95=[round(lo, 3), round(hi, 3)],
                          grasped=sum(r["grasped"] for r in R), on_goal=sum(r["on_goal"] for r in R),
                          bottle_collision=sum(r["bottle_collision"] for r in R))
        rows.append(row)
    (eval_dir / "summary.json").write_text(json.dumps(rows, indent=1))
    lines = ["| Training data | Rollouts | Clean scenes | Blocked-route scenes (bottle hits) |", "|---|---|---|---|"]
    for r in rows:
        cell = lambda s: (f"{r[s]['success']}/{r[s]['n']} ({r[s]['rate'] * 100:.0f}%, CI {r[s]['ci95'][0] * 100:.0f}–{r[s]['ci95'][1] * 100:.0f})"  # noqa: E731
                          + (f", {r[s]['bottle_collision']} hits" if s == "blocked" else "")) if s in r else "–"
        lines.append(f"| {r['label']} | {r['rollouts']} | {cell('clean')} | {cell('blocked')} |")
    (eval_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
