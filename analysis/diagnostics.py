"""Diagnostics for one model run (CPU only).

D1  Are clusters just position buckets? Predict a step's cluster from its
    absolute and relative step index (gradient boosting, 5-fold CV grouped
    by trace) and compare with the majority-class rate.
D2  Do incorrect traces self-loop more? Per-trace self-loop rate
    P(z_i == z_{i-1}) for correct vs incorrect traces, raw and averaged over
    within-bin differences after binning traces by step count (quartiles).
D3  Is the steering effect a discourse-marker swap? For the held-out eval
    results: distribution of the regenerated step's first word per condition,
    exit rate within each marker, and SOPHIA's exit rate re-weighted to the
    greedy condition's marker mix. No-marker generations count as not exiting.

Usage:
  python analysis/diagnostics.py --run-dir runs/qwen3_4b [--rule prev_same]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

DATASETS = ("gsm8k", "aqua", "logiqa", "math")
CONDS = ("none", "random", "negative", "global", "steered")


def load_clustered(run: Path, ds: str):
    return [json.loads(l) for l in (run / "clustered" / f"{ds}_clustered.jsonl").open()]


def position_test(run: Path):
    print("\n=== D1: cluster predictability from step position (5-fold CV grouped by trace) ===")
    print(f"{'dataset':8s} {'majority':>8s} {'abs_idx':>8s} {'rel_idx':>8s} {'abs+rel':>8s}")
    for ds in DATASETS:
        X, y, g = [], [], []
        for r in load_clustered(run, ds):
            sl = r.get("state_labels") or []
            n = len(sl)
            for i, c in enumerate(sl):
                X.append([i, i / max(n - 1, 1)])
                y.append(c)
                g.append(r["sample_id"])
        X, y, g = np.array(X), np.array(y), np.array(g)
        majority = Counter(y).most_common(1)[0][1] / len(y)
        scores = []
        for cols in ([0], [1], [0, 1]):
            clf = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=0)
            scores.append(cross_val_score(clf, X[:, cols], y, groups=g, cv=GroupKFold(5)).mean())
        print(f"{ds:8s} {majority:8.3f} {scores[0]:8.3f} {scores[1]:8.3f} {scores[2]:8.3f}")


def selfloop_test(run: Path):
    print("\n=== D2: per-trace self-loop rate P(z_i == z_{i-1}), correct vs incorrect ===")
    for ds in DATASETS:
        stats = {True: [], False: []}
        for r in load_clustered(run, ds):
            sl = r.get("state_labels") or []
            if len(sl) < 3:
                continue
            loops = sum(sl[i] == sl[i - 1] for i in range(1, len(sl)))
            stats[bool(r["is_correct"])].append((len(sl), loops / (len(sl) - 1)))
        c, w = np.array(stats[True]), np.array(stats[False])
        if len(c) == 0 or len(w) == 0:
            print(f"{ds}: n_correct={len(c)} n_incorrect={len(w)} (skipped)")
            continue
        edges = np.quantile(np.concatenate([c[:, 0], w[:, 0]]), [0, .25, .5, .75, 1.0])
        diffs = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            cm = c[(c[:, 0] >= lo) & (c[:, 0] <= hi), 1]
            wm = w[(w[:, 0] >= lo) & (w[:, 0] <= hi), 1]
            if len(cm) >= 5 and len(wm) >= 5:
                diffs.append(wm.mean() - cm.mean())
        binned = f"{np.mean(diffs):+.3f} over {len(diffs)} bins" if diffs else "n/a (too few traces per bin)"
        print(f"{ds:8s} n_correct={len(c):3d} n_incorrect={len(w):3d}  "
              f"steps correct/incorrect={c[:, 0].mean():5.1f}/{w[:, 0].mean():5.1f}  "
              f"self-loop correct/incorrect={c[:, 1].mean():.3f}/{w[:, 1].mean():.3f}  "
              f"length-binned diff (incorrect - correct)={binned}")


def first_word(text: str) -> str:
    m = re.match(r"\s*([A-Za-z']+)", text or "")
    return m.group(1).lower() if m else "<none>"


def marker_test(run: Path, rule: str):
    rows = []
    for ds in DATASETS:
        p = run / "crossfit" / ds / f"hit_rate_results_{rule}_a0.05_seed42.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.open()]
    print(f"\n=== D3: opening word of the regenerated step by condition ({rule}, {len(rows)} rows) ===")
    if not rows:
        print("no eval results found")
        return
    count = defaultdict(Counter)
    exits = defaultdict(lambda: defaultdict(list))
    for r in rows:
        w = first_word(r.get("regen_text"))
        count[r["condition"]][w] += 1
        exits[r["condition"]][w].append(bool(r["crossed"]))
    conds = [c for c in CONDS if c in count]
    n = {c: sum(count[c].values()) for c in conds}
    top = [w for w, _ in (count["none"] + count["steered"]).most_common(6)]
    print(f"{'word':14s}" + "".join(f"{c:>22s}" for c in conds))
    for w in top:
        cells = []
        for c in conds:
            e = exits[c][w]
            rate = f"{100 * np.mean(e):3.0f}%" if e else "  -"
            cells.append(f"{100 * count[c][w] / n[c]:5.1f}% exit={rate}   ")
        print(f"{w:14s}" + "".join(cells))
    words = set().union(*[count[c].keys() for c in conds])
    for c in conds:
        if c != "none":
            tv = 0.5 * sum(abs(count[c][w] / n[c] - count["none"][w] / n["none"]) for w in words)
            print(f"  word-distribution TV distance, {c} vs greedy: {tv:.3f}")
    num = den = 0.0
    for w in words:
        if exits["none"][w] and exits["steered"][w]:
            num += count["none"][w] * np.mean(exits["steered"][w])
            den += count["none"][w]
    overall = {c: round(float(np.mean([bool(r["crossed"]) for r in rows if r["condition"] == c])), 3)
               for c in conds}
    print(f"  overall exit rate: {overall}")
    if den:
        print(f"  SOPHIA exit rate re-weighted to greedy's word mix: {num / den:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--rule", default="prev_same", choices=("prev_same", "next_same"))
    ap.add_argument("--skip", default="", help="comma list of tests to skip: position,selfloop,marker")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    if "position" not in skip:
        position_test(args.run_dir)
    if "selfloop" not in skip:
        selfloop_test(args.run_dir)
    if "marker" not in skip:
        marker_test(args.run_dir, args.rule)


if __name__ == "__main__":
    main()
