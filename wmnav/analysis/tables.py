"""Collect evaluation summaries into result tables.

    python -m wmnav.analysis.tables RUNS_DIR [--out results] [--by name]

Reads every ``summary.json`` under RUNS_DIR written by ``wmnav.evaluate`` or
``wmnav.eval_ppo`` and writes:

* ``<out>.csv``: one row per run (model, level, panels, games, wins, crashes,
  timeouts, rates, binomial standard error of the win rate);
* ``<out>_<metric>_p<panels>.csv``: model x level matrices per panel count
  (repeated runs of a cell pooled by games), with row and column means.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def collect(root: Path, by: str = "name") -> pd.DataFrame:
    """One row per evaluation run found under ``root`` (any ``summary.json`` with a level
    and a game count), with the win rate's binomial standard error. ``by`` picks
    the summary field used as the model label.
    """
    rows = []
    for f in sorted(Path(root).rglob("summary.json")):
        s = json.loads(f.read_text())
        if "level" not in s or "games" not in s:
            continue
        model = s.get(by) or s.get("experiment") or s.get("name") or f.parent.parent.name
        games = s["games"]
        win = s.get("win_rate")
        rows.append({
            "model": model, "level": s["level"], "panels": s.get("num_panels"), "games": games,
            "win_rate": win, "crash_rate": s.get("crash_rate"), "timeout_rate": s.get("timeout_rate"),
            "win_se": math.sqrt(win * (1 - win) / games) if games and win is not None else float("nan"),
            "run_dir": str(f.parent),
        })
    return pd.DataFrame(rows)


def matrices(df: pd.DataFrame, metric: str):
    """model x level matrix of ``metric`` per panel count. Several runs of the same
    cell are pooled by games (sum of events / sum of games), not averaged."""
    out = {}
    df = df.assign(_events=df[metric] * df["games"])
    for panels, g in df.groupby("panels"):
        cells = g.groupby(["model", "level"])[["_events", "games"]].sum()
        m = (cells["_events"] / cells["games"]).unstack("level")
        m["mean"] = m.mean(axis=1)
        m.loc["mean"] = m.mean(axis=0)
        out[panels] = m
    return out


def main():
    """Command-line entry point: print and optionally save the result tables."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", default="results")
    ap.add_argument("--by", default="name", help="summary field naming the model (name | experiment)")
    args = ap.parse_args()
    df = collect(args.root, args.by)
    if df.empty:
        raise SystemExit(f"no evaluation summaries under {args.root}")
    df.sort_values(["panels", "model", "level"]).to_csv(f"{args.out}.csv", index=False)
    for metric in ("win_rate", "crash_rate", "timeout_rate"):
        for panels, m in matrices(df, metric).items():
            m.round(4).to_csv(f"{args.out}_{metric}_p{panels}.csv")
            if metric == "win_rate":
                print(f"\nwin rate, {panels} panels\n{m.round(3).to_string()}")


if __name__ == "__main__":
    main()
