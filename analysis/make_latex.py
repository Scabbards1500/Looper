"""LaTeX tables for the held-out prev_same results across Qwen3 scale.

Table 1 (per cluster) follows the paper's tab:results_per_cluster layout;
table 2 gives per-dataset and pooled rates with paired exact McNemar tests.
Rates use conservative coding: a generation with no step boundary within 200
tokens counts as not exiting, so every selected prefix is in the denominator.
Cluster columns are the clusters with steering vectors, renumbered C0..Ck in
order of their K-means id within each (model, dataset).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

DATASETS = ("gsm8k", "aqua", "logiqa", "math")
ROWS = [("none", "Greedy decoding"), ("random", "Random steering"),
        ("negative", "Negative steering"), ("global", "Global vector"), ("steered", "SOPHIA")]
RESULT = "hit_rate_results_{rule}_a0.05_seed42.jsonl"
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def load(cf_dir, rule):
    out = {}  # (ds, sid, step) -> {"cluster": c, cond: exited}
    for ds in DATASETS:
        for line in (cf_dir / ds / RESULT.format(rule=rule)).open():
            r = json.loads(line)
            d = out.setdefault((ds, r["sample_id"], r["step_idx"]), {"cluster": r["cluster"]})
            d[r["condition"]] = bool(r["crossed"])
    return out


def fmt_col(vals):
    """Bold the best value(s), underline the second-best distinct value; no
    marks when every method ties."""
    distinct = sorted(set(vals), reverse=True)
    if len(distinct) == 1:
        return [f"{v:.1f}" for v in vals]
    best, second = distinct[0], distinct[1]
    return [f"\\textbf{{{v:.1f}}}" if v == best else
            f"\\underline{{{v:.1f}}}" if v == second else f"{v:.1f}" for v in vals]


def stars(p):
    return "^{***}" if p < 1e-3 else "^{**}" if p < 1e-2 else "^{*}" if p < 0.05 else ""


def mcnemar(rows, a, b):
    x = sum(r[a] and not r[b] for r in rows)
    y = sum(r[b] and not r[a] for r in rows)
    return binomtest(x, x + y, 0.5).pvalue if x + y else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True,
                    help="LABEL=RUN_DIR (uses RUN_DIR/crossfit); repeat in table order")
    ap.add_argument("--rule", default="prev_same", choices=("prev_same", "next_same"))
    ap.add_argument("--out", type=Path, default=Path("reports/tables_scale.tex"))
    args = ap.parse_args()
    MODELS = []
    for spec in args.model:
        label, _, run = spec.partition("=")
        if not run:
            raise SystemExit(f"--model expects LABEL=RUN_DIR, got {spec!r}")
        MODELS.append((label, Path(run) / "crossfit"))
    data = {m: load(cf, args.rule) for m, cf in MODELS}
    # Cluster columns per (model, dataset), renumbered by K-means id.
    clusters = {(m, ds): sorted({v["cluster"] for k, v in res.items() if k[0] == ds})
                for m, res in data.items() for ds in DATASETS}
    ncol = {ds: max(len(clusters[(m, ds)]) for m, _ in MODELS) for ds in DATASETS}
    small_n = []

    # ---------- Table 1: per cluster ----------
    spec = "l " + " ".join("c" * ncol[ds] for ds in DATASETS)
    total = 1 + sum(ncol.values())
    L = [r"\begin{table*}[t]", r"    \centering",
         r"    \caption{Per-cluster loop-exit rate (\%) under held-out 5-fold trace-level",
         f"cross-fitting, across four reasoning datasets and {NUM_WORDS.get(len(MODELS), len(MODELS))} steering targets.",
         r"Each cell uses up to $N{=}30$ stayer prefixes ($z_{i-1}{=}z_i$) from held-out",
         r"incorrect traces; clustering, z-scoring and all steering vectors are fit only",
         r"on the other folds. A regenerated step with no step boundary within 200 tokens",
         r"counts as not exiting. \sophia{} patches the next step's residual stream with",
         r"$+\alpha v_{c}^{(\ell)}$; \textbf{Negative steering} uses $-\alpha v_{c}^{(\ell)}$;",
         r"\textbf{Random steering} is a norm-matched Gaussian; \textbf{Global vector} is a",
         r"single crosser-minus-stayer direction pooled over all clusters, scaled",
         r"identically; \textbf{Greedy decoding} is the unintervened baseline. Cluster ids",
         r"are dataset- and model-local; the intro-tag cluster has no vector and is",
         r"omitted. $\alpha{=}0.05$ for all models (not retuned); $\mathcal{L}$ is the upper",
         r"half of the decoder. Bold: best per column; underline: second best.",
         "SMALLN}",
         r"    \label{tab:results_per_cluster}",
         r"    \setlength{\tabcolsep}{4pt}", r"    \renewcommand{\arraystretch}{1.2}",
         r"\resizebox{\textwidth}{!}{%", f"    \\begin{{tabular}}{{{spec}}}", r"    \toprule",
         "     & " + " & ".join(f"\\multicolumn{{{ncol[ds]}}}{{c}}{{{ds}}}" for ds in DATASETS) + r" \\"]
    start, cm = 2, []
    for ds in DATASETS:
        cm.append(f"\\cmidrule(lr){{{start}-{start + ncol[ds] - 1}}}")
        start += ncol[ds]
    L.append("    " + " ".join(cm))
    L.append("    Method & " + " & ".join(f"C{j}" for ds in DATASETS for j in range(ncol[ds])) + r" \\")
    for m, _ in MODELS:
        res = data[m]
        L += [r"    \midrule", f"    \\multicolumn{{{total}}}{{l}}{{\\textit{{{m}}}}} \\\\", r"    \midrule"]
        cols = []  # list of per-column {cond: rate} or None
        for ds in DATASETS:
            cl = clusters[(m, ds)]
            for j in range(ncol[ds]):
                if j >= len(cl):
                    cols.append(None)
                    continue
                rows = [v for k, v in res.items() if k[0] == ds and v["cluster"] == cl[j]]
                if len(rows) < 30:
                    small_n.append(f"{m} {ds} C{j} ($N{{=}}{len(rows)}$)")
                cols.append({c: 100 * sum(r[c] for r in rows) / len(rows) for c, _ in ROWS})
        cells = [fmt_col([col[c] for c, _ in ROWS]) if col else ["--"] * len(ROWS) for col in cols]
        for i, (c, name) in enumerate(ROWS):
            if c == "steered":
                L.append(r"    \rowcolor{sealblue}")
            L.append(f"    {name:<17} & " + " & ".join(cell[i] for cell in cells) + r" \\")
    L += [r"    \bottomrule", r"    \end{tabular}%", r"    }", r"\end{table*}"]
    note = (" Cells with fewer prefixes: " + "; ".join(small_n) + ".") if small_n else ""
    missing = [f"{m} {ds}" for m, _ in MODELS for ds in DATASETS if len(clusters[(m, ds)]) < ncol[ds]]
    if missing:
        note += " ``--'': the model has fewer clusters with vectors on that dataset."
    table1 = "\n".join(L).replace("SMALLN}", note.strip() and note + "}" or "}")

    # ---------- Table 2: per dataset + pooled, paired tests ----------
    T = [r"\begin{table}[t]", r"    \centering",
         r"    \caption{Held-out loop-exit rate (\%) per dataset and pooled over all",
         r"selected prefixes, same protocol as Table~\ref{tab:results_per_cluster}.",
         r"$\Delta$ rows give \sophia{} minus the baseline in percentage points; stars",
         r"mark paired exact McNemar tests on the same prefixes",
         r"($^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$). $N$ is the number of",
         r"evaluated prefixes.}",
         r"    \label{tab:results_scale}",
         r"    \setlength{\tabcolsep}{4pt}", r"    \renewcommand{\arraystretch}{1.15}",
         r"\resizebox{\columnwidth}{!}{%", r"    \begin{tabular}{l ccccc}", r"    \toprule",
         r"    Method & gsm8k & aqua & logiqa & math & Pooled \\"]
    for m, _ in MODELS:
        res = data[m]
        groups = [[v for k, v in res.items() if k[0] == ds] for ds in DATASETS] + [list(res.values())]
        ns = " / ".join(str(len(g)) for g in groups)
        T += [r"    \midrule", f"    \\multicolumn{{6}}{{l}}{{\\textit{{{m}}} ($N$ = {ns})}} \\\\", r"    \midrule"]
        rates = [{c: 100 * sum(r[c] for r in g) / len(g) for c, _ in ROWS} for g in groups]
        cells = [fmt_col([rt[c] for c, _ in ROWS]) for rt in rates]
        for i, (c, name) in enumerate(ROWS):
            if c == "steered":
                T.append(r"    \rowcolor{sealblue}")
            T.append(f"    {name:<17} & " + " & ".join(cell[i] for cell in cells) + r" \\")
        for base, label in (("none", r"$\Delta$ vs.\ Greedy"), ("global", r"$\Delta$ vs.\ Global")):
            vals = []
            for g, rt in zip(groups, rates):
                d = rt["steered"] - rt[base]
                vals.append(f"${d:+.1f}{stars(mcnemar(g, 'steered', base))}$")
            T.append(f"    {label:<17} & " + " & ".join(vals) + r" \\")
    T += [r"    \bottomrule", r"    \end{tabular}%", r"    }", r"\end{table}"]
    table2 = "\n".join(T)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"% Generated by analysis/make_latex.py from the {args.rule} cross-fit results.\n"
                   "% Needs \\usepackage{booktabs,colortbl} and the paper's \\sophia, sealblue.\n\n"
                   + table1 + "\n\n" + table2 + "\n")
    print(table1 + "\n\n" + table2)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
