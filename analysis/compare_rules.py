"""Compare held-out results under the two stayer-selection rules.

next_same: prefix through step i with z_i == z_{i+1} in the original trace
           (what the paper's eval code did).
prev_same: prefix through step i with z_{i-1} == z_i (the online controller's
           self-loop gate, and what the paper's text describes).

Conservative coding throughout: a generation with no step boundary within
200 tokens ("no marker") counts as not exiting, and every selected prefix is
in the denominator. Significance: exact McNemar on paired outcomes (each
prefix is decoded under every condition). Rule-vs-rule differences use
Fisher's exact test on the pooled 2x2 table; the two prefix samples are
almost disjoint (shared prefixes are reported), so treating them as
independent is a close approximation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

DATASETS = ("gsm8k", "aqua", "logiqa", "math")
RULES = ("next_same", "prev_same")
CONDS = ("none", "random", "negative", "global", "steered")
LABEL = {"none": "Greedy", "random": "Random", "negative": "Negative",
         "global": "Global", "steered": "SOPHIA"}


def load(work, rule):
    """{(ds, sid, step): {cond: exited(bool)}} plus per-prefix cluster."""
    out, cluster = {}, {}
    for ds in DATASETS:
        p = work / ds / f"hit_rate_results_{rule}_a0.05_seed42.jsonl"
        with p.open() as f:
            for line in f:
                r = json.loads(line)
                key = (ds, r["sample_id"], r["step_idx"])
                out.setdefault(key, {})[r["condition"]] = bool(r["crossed"])
                cluster[key] = r["cluster"]
    incomplete = [k for k, v in out.items() if set(v) != set(CONDS)]
    if incomplete:
        raise SystemExit(f"{rule}: {len(incomplete)} prefixes missing conditions, e.g. {incomplete[:3]}")
    return out, cluster


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(res, keys, a, b):
    """Exact McNemar of condition a vs b over prefixes `keys`."""
    only_a = sum(res[k][a] and not res[k][b] for k in keys)
    only_b = sum(res[k][b] and not res[k][a] for k in keys)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else float("nan")
    return only_a, only_b, p


def subsets(res):
    yield from ((ds, [k for k in res if k[0] == ds]) for ds in DATASETS)
    yield "Pooled", list(res)


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="one model's cross-fit dir with results for both rules")
    ap.add_argument("--out", type=Path, default=Path("reports/compare_rules.md"))
    args = ap.parse_args()
    data = {rule: load(args.work_dir, rule) for rule in RULES}
    lines = []
    emit = lines.append

    for rule in RULES:
        res, cluster = data[rule]
        emit(f"\n## {rule}: held-out exit rate (%), no-marker = stay\n")
        emit("| dataset | prefixes | " + " | ".join(LABEL[c] for c in CONDS) + " |")
        emit("|---|---|" + "---|" * len(CONDS))
        for name, keys in subsets(res):
            n = len(keys)
            cells = []
            for c in CONDS:
                k = sum(res[x][c] for x in keys)
                cell = f"{100 * k / n:.1f}"
                if name == "Pooled":
                    lo, hi = wilson(k, n)
                    cell += f" [{lo:.1f}, {hi:.1f}]"
                cells.append(cell)
            b = "**" if name == "Pooled" else ""
            emit(f"| {b}{name}{b} | {n} | " + " | ".join(cells) + " |")

        emit(f"\nPaired exact McNemar ({rule}); counts are prefixes where only one side exited:\n")
        emit("| dataset | SOPHIA vs Global: SOPHIA-only / Global-only | p | "
             "SOPHIA vs Greedy: SOPHIA-only / Greedy-only | p |")
        emit("|---|---|---|---|---|")
        for name, keys in subsets(res):
            g = mcnemar(res, keys, "steered", "global")
            n0 = mcnemar(res, keys, "steered", "none")
            emit(f"| {name} | {g[0]} / {g[1]} | {fmt_p(g[2])} | {n0[0]} / {n0[1]} | {fmt_p(n0[2])} |")

    # Rule sensitivity.
    a, _ = data["next_same"]
    b, _ = data["prev_same"]
    shared = sorted(set(a) & set(b))
    emit("\n## Rule sensitivity: prev_same minus next_same (percentage points)\n")
    emit("| dataset | " + " | ".join(LABEL[c] for c in CONDS) +
         " | SOPHIA−Global (next → prev) | SOPHIA−Greedy (next → prev) |")
    emit("|---|" + "---|" * (len(CONDS) + 2))
    for ds in DATASETS + ("Pooled",):
        ka = [k for k in a if ds == "Pooled" or k[0] == ds]
        kb = [k for k in b if ds == "Pooled" or k[0] == ds]
        rate = lambda res, keys, c: 100 * sum(res[k][c] for k in keys) / len(keys)
        cells = []
        for c in CONDS:
            ra, rb = rate(a, ka, c), rate(b, kb, c)
            xa, xb = sum(a[k][c] for k in ka), sum(b[k][c] for k in kb)
            p = fisher_exact([[xa, len(ka) - xa], [xb, len(kb) - xb]])[1]
            cells.append(f"{rb - ra:+.1f} (p={p:.2f})")
        gap = lambda res, keys, other: rate(res, keys, "steered") - rate(res, keys, other)
        cells.append(f"{gap(a, ka, 'global'):+.1f} → {gap(b, kb, 'global'):+.1f}")
        cells.append(f"{gap(a, ka, 'none'):+.1f} → {gap(b, kb, 'none'):+.1f}")
        emit(f"| {ds} | " + " | ".join(cells) + " |")

    agree = sum(a[k][c] == b[k][c] for k in shared for c in CONDS)
    emit(f"\nShared prefixes between the two samples: {len(shared)} of {len(a)} / {len(b)}. "
         f"Outcome agreement on shared prefixes across the two runs: "
         f"{agree}/{len(shared) * len(CONDS)} (determinism check).")

    text = "\n".join(lines)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
