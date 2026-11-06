"""Cross-scale report for the held-out prev_same SOPHIA evaluation.

Per model: prefixes, conservative exit rates (no-marker = stay) for every
condition, SOPHIA vs Greedy and SOPHIA vs Global (difference, exact McNemar),
pooled results, fold clustering stability, per-fold vs full-data vector
cosine, no-marker rates and trace statistics. Then a compact cross-scale table
and the four requested flags.

Flag thresholds: SOPHIA "does not beat" a baseline when the difference is <= 0
or p >= 0.05; clustering is "unstable" when any fold's held-out label
agreement with the full-data partition is < 0.90; no-marker frequency changes
"substantially" when a condition's rate on a dataset moves >= 10 percentage
points relative to the --reference model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import binomtest

DATASETS = ("gsm8k", "aqua", "logiqa", "math")
CONDS = ("none", "random", "negative", "global", "steered")
LABEL = {"none": "Greedy", "random": "Random", "negative": "Negative", "global": "Global", "steered": "SOPHIA"}
RESULT = "hit_rate_results_{rule}_a0.05_seed42.jsonl"


def parse_models(specs):
    """--model LABEL=RUN_DIR, where RUN_DIR holds crossfit/ and clustered/."""
    out = []
    for spec in specs:
        label, _, run = spec.partition("=")
        if not run:
            raise SystemExit(f"--model expects LABEL=RUN_DIR, got {spec!r}")
        out.append((label, Path(run) / "crossfit", Path(run) / "clustered"))
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(res, keys, a, b):
    only_a = sum(res[k][a] and not res[k][b] for k in keys)
    only_b = sum(res[k][b] and not res[k][a] for k in keys)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def fp(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def load_model(cf_dir: Path, cl_dir: Path, rule: str):
    res, nomark = {}, {}
    for ds in DATASETS:
        path = cf_dir / ds / RESULT.format(rule=rule)
        if not path.exists():
            return None
        for line in path.open():
            r = json.loads(line)
            key = (ds, r["sample_id"], r["step_idx"])
            res.setdefault(key, {})[r["condition"]] = bool(r["crossed"])
            nomark.setdefault((ds, r["condition"]), []).append(r["crossed"] is None)
    bad = [k for k, v in res.items() if set(v) != set(CONDS)]
    if bad:
        raise SystemExit(f"{cf_dir}: {len(bad)} prefixes missing conditions")
    builds = {ds: json.load((cf_dir / ds / "build_summary.json").open()) for ds in DATASETS}
    traces = {}
    for ds in DATASETS:
        recs = [json.loads(l) for l in (cl_dir / f"{ds}_clustered.jsonl").open()]
        traces[ds] = {"n": len(recs), "acc": sum(r["is_correct"] for r in recs) / len(recs),
                      "n_incorrect": sum(not r["is_correct"] for r in recs),
                      "steps": sum(len(r["steps"]) for r in recs) / len(recs),
                      "tokens": sum(r.get("n_generated_tokens", 0) for r in recs) / len(recs)}
    return res, nomark, builds, traces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="LABEL=RUN_DIR, repeat in the order to report (e.g. by scale)")
    ap.add_argument("--reference", default=None,
                    help="label used as the no-marker reference (default: first --model)")
    ap.add_argument("--rule", default="prev_same", choices=("prev_same", "next_same"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()
    MODELS = parse_models(args.model)
    ref = args.reference or MODELS[0][0]
    out, flags, summary = [], [], {}
    emit = out.append
    loaded = {}
    for label, cf, cl in MODELS:
        m = load_model(cf, cl, args.rule)
        if m is None:
            emit(f"\n# {label}: results not complete yet\n")
            continue
        loaded[label] = m
    ref_nomark = None
    if ref in loaded:
        nm = loaded[ref][1]
        ref_nomark = {k: 100 * sum(v) / len(v) for k, v in nm.items()}

    for label, (res, nomark, builds, traces) in loaded.items():
        emit(f"\n# {label}\n")
        emit("Traces (500/dataset): accuracy, incorrect traces (the prefix pool), mean steps, mean tokens\n")
        emit("| dataset | accuracy | incorrect | steps/trace | tokens/trace |\n|---|---|---|---|---|")
        for ds in DATASETS:
            t = traces[ds]
            emit(f"| {ds} | {100*t['acc']:.1f}% | {t['n_incorrect']} | {t['steps']:.1f} | {t['tokens']:.0f} |")

        emit("\nHeld-out exit rate (%), prev_same, no-marker = stay\n")
        emit("| dataset | prefixes | " + " | ".join(LABEL[c] for c in CONDS) +
             " | SOPHIA−Greedy (p) | SOPHIA−Global (p) |")
        emit("|---|---|" + "---|" * (len(CONDS) + 2))
        summary[label] = {}
        for name in DATASETS + ("Pooled",):
            keys = [k for k in res if name == "Pooled" or k[0] == name]
            n = len(keys)
            rate = {c: 100 * sum(res[k][c] for k in keys) / n for c in CONDS}
            g = mcnemar(res, keys, "steered", "none")
            gl = mcnemar(res, keys, "steered", "global")
            d_g, d_gl = rate["steered"] - rate["none"], rate["steered"] - rate["global"]
            summary[label][name] = {"n": n, **rate, "d_greedy": d_g, "p_greedy": g[2],
                                    "d_global": d_gl, "p_global": gl[2], "mc_greedy": g[:2], "mc_global": gl[:2]}
            cells = []
            for c in CONDS:
                cell = f"{rate[c]:.1f}"
                if name == "Pooled":
                    lo, hi = wilson(sum(res[k][c] for k in keys), n)
                    cell += f" [{lo:.1f}, {hi:.1f}]"
                cells.append(cell)
            b = "**" if name == "Pooled" else ""
            emit(f"| {b}{name}{b} | {n} | " + " | ".join(cells) +
                 f" | {d_g:+.1f} ({fp(g[2])}; {g[0]}/{g[1]}) | {d_gl:+.1f} ({fp(gl[2])}; {gl[0]}/{gl[1]}) |")
            if d_g <= 0 or g[2] >= 0.05:
                flags.append(f"[1] {label} {name}: SOPHIA does not beat Greedy ({d_g:+.1f} pt, p={fp(g[2])})")
            if d_gl <= 0:
                sig = " (significant)" if gl[2] < 0.05 else ""
                flags.append(f"[2] {label} {name}: Global ≥ SOPHIA ({-d_gl:+.1f} pt for Global, p={fp(gl[2])}){sig}")

        emit("\nClustering stability across folds (fold K-means vs full-data partition) and "
             "per-fold vs full-data steering-vector cosine (mean over layers)\n")
        emit("| dataset | train-step match: min / mean | held-out agreement: min / mean | "
             "vector cosine: min / mean | unstable folds |")
        emit("|---|---|---|---|---|")
        for ds in DATASETS:
            pf = builds[ds]["per_fold"]
            mt = [f["match_rate_train_vs_full"] for f in pf]
            ag = [f["heldout_agreement_vs_full"] for f in pf]
            cs = [v for f in pf for v in f["mean_cos_vs_full_vector"].values()]
            unstable = [f["fold"] for f in pf if f["heldout_agreement_vs_full"] < 0.90]
            summary[label].setdefault("stability", {})[ds] = (min(ag), sum(ag) / len(ag), min(cs), sum(cs) / len(cs))
            emit(f"| {ds} | {min(mt):.3f} / {sum(mt)/len(mt):.3f} | {min(ag):.3f} / {sum(ag)/len(ag):.3f} | "
                 f"{min(cs):.3f} / {sum(cs)/len(cs):.3f} | {unstable or '–'} |")
            if unstable:
                flags.append(f"[3] {label} {ds}: clustering unstable in folds {unstable} "
                             f"(held-out agreement min {min(ag):.3f}); vector cosine min {min(cs):.3f}")

        emit("\nNo-marker rate (% of prefixes with no step boundary within 200 tokens)\n")
        emit("| dataset | " + " | ".join(LABEL[c] for c in CONDS) + " |")
        emit("|---|" + "---|" * len(CONDS))
        for ds in DATASETS:
            vals = []
            for c in CONDS:
                v = nomark[(ds, c)]
                r = 100 * sum(v) / len(v)
                vals.append(f"{r:.1f}")
                if ref_nomark and label != ref and abs(r - ref_nomark[(ds, c)]) >= 10:
                    flags.append(f"[4] {label} {ds} {LABEL[c]}: no-marker {r:.1f}% vs {ref_nomark[(ds, c)]:.1f}% on {ref}")
            emit(f"| {ds} | " + " | ".join(vals) + " |")

    if len(summary) > 1:
        emit("\n# Cross-scale comparison (held-out, prev_same, no-marker = stay)\n")
        emit("| model | dataset | n | Greedy | Global | SOPHIA | SOPHIA−Greedy (p) | SOPHIA−Global (p) |")
        emit("|---|---|---|---|---|---|---|---|")
        for name in DATASETS + ("Pooled",):
            for label in summary:
                s = summary[label][name]
                b = "**" if name == "Pooled" else ""
                emit(f"| {label} | {b}{name}{b} | {s['n']} | {s['none']:.1f} | {s['global']:.1f} | {s['steered']:.1f} | "
                     f"{s['d_greedy']:+.1f} ({fp(s['p_greedy'])}) | {s['d_global']:+.1f} ({fp(s['p_global'])}) |")

    emit("\n# Flags\n")
    emit("\n".join(f"- {f}" for f in flags) if flags else "- none")
    text = "\n".join(out)
    print(text)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"report_scale_{args.rule}.md").write_text(text + "\n")
    (args.out_dir / f"report_scale_{args.rule}.json").write_text(json.dumps(summary, indent=2, default=list))


if __name__ == "__main__":
    main()
