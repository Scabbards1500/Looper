"""Held-out hit-rate evaluation of SOPHIA under trace-level cross-fitting.

For each selected stayer prefix the next reasoning step is regenerated
greedily under five conditions and re-clustered; a hit is an exit from the
prefix's cluster. Generation, prefix building and re-clustering come from
sophia.eval_core.

  - Stayer prefixes come from heldout_labels.jsonl: each trace is labelled by
    the fold whose fit excluded it, and that fold's z-score stats, centroids
    and vectors are used for both patching and re-clustering that prefix.
  - Conditions: none (greedy), steered (SOPHIA, +alpha v_c), negative
    (-alpha v_c), random (norm-matched Gaussian), global (one pooled
    crosser-minus-stayer direction per fold, normalised like "steered").
  - --stayer-rule prev_same selects prefixes with z_{i-1} == z_i (the online
    controller's self-loop gate); next_same selects z_i == z_{i+1}.
  - Random-control seeds use crc32, so the control is reproducible.
  - Appends to the output file and skips (sample_id, step_idx, condition)
    rows already written, so a killed run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sophia.eval_core import (
    BASE_MODEL, THINKING_MODEL, build_instruct_prefix,
    cluster_step_meanpool, first_token_at_or_after, parse_layers, regen_one,
    strip_step_prefix,
)

CONDITIONS = ("none", "steered", "negative", "random", "global")


class BackboneOnly:
    """Stand-in for the base LM inside cluster_step_meanpool, which only reads
    hidden_states[-1]. The backbone's last_hidden_state is bit-identical to it
    (checked on a 2.3k-token math trace: max abs diff 0.0) but skips the
    full-sequence LM-head logits, the allocation that OOMed on shared GPUs."""

    def __init__(self, lm):
        self.lm = lm

    def __call__(self, output_hidden_states=True, return_dict=True, **enc):
        return SimpleNamespace(hidden_states=(self.lm.model(**enc).last_hidden_state,))


class ThinkOpenTemplate:
    """Tokenizer stand-in for build_instruct_prefix. The Qwen3-4B-Thinking-2507
    generation prompt ends with "<think>\\n"; hybrid Qwen3 templates (1.7B, 8B)
    end at "assistant\\n". Appending "<think>\\n" when missing gives every model
    the same prefix structure (chat prompt, <think>, steps); no-op for 4B."""

    def __init__(self, tok):
        self.tok = tok

    def apply_chat_template(self, *a, **kw):
        s = self.tok.apply_chat_template(*a, **kw)
        return s if s.endswith("<think>\n") else s + "<think>\n"


def select_stayers(heldout, vec_clusters_by_fold, n_steps_by_sid, n_per_cluster,
                   seed, rule, k=5):
    """next_same (paper code): z_i == z_{i+1} in the original trace.
    prev_same (paper text / online gate): z_{i-1} == z_i, i-1 >= 1."""
    pool = defaultdict(list)
    for sid, row in heldout.items():
        if row["is_correct"]:
            continue
        sl = row["state_labels"]
        n = n_steps_by_sid[sid]
        for i in range(1, min(len(sl), n) - 1):
            c = sl[i]
            if c < 0 or c not in vec_clusters_by_fold[row["fold"]]:
                continue
            if rule == "next_same" and sl[i + 1] == c:
                pool[c].append((sid, i))
            elif rule == "prev_same" and i >= 2 and sl[i - 1] == c:
                pool[c].append((sid, i))
    rng = np.random.default_rng(seed)
    selected = {}
    for c in range(k):
        idxs = pool.get(c, [])
        rng.shuffle(idxs)
        selected[c] = idxs[:n_per_cluster]
    return selected, {c: len(pool.get(c, [])) for c in range(k)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="cross-fit directory written by crossfit_build (<run>/crossfit)")
    ap.add_argument("--n-per-cluster", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch-layers", default="18-35")
    ap.add_argument("--stayer-rule", default="next_same", choices=("next_same", "prev_same"))
    ap.add_argument("--device-thinking", default="cuda:0")
    ap.add_argument("--device-base", default=None, help="defaults to --device-thinking")
    ap.add_argument("--tag", default="")
    ap.add_argument("--target-model", default=THINKING_MODEL)
    ap.add_argument("--clustered-dir", type=Path, required=True,
                    help="<run>/clustered, source of the trace text")
    args = ap.parse_args()
    dev_t = args.device_thinking
    dev_b = args.device_base or dev_t
    ds = args.dataset
    ds_dir = args.work_dir / ds
    patch_layers = parse_layers(args.patch_layers)

    folds = {}
    for p in sorted(ds_dir.glob("fold*.pt")):
        a = torch.load(p, map_location="cpu", weights_only=False)
        folds[a["fold"]] = a
    if not folds:
        raise SystemExit(f"no fold artifacts in {ds_dir}; run crossfit_build.py first")
    layers = folds[0]["layers"]
    missing = [li for li in patch_layers if li not in layers]
    if missing:
        raise SystemExit(f"vectors missing for layers {missing}")

    records = {}
    with (args.clustered_dir / f"{ds}_clustered.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            records[r["sample_id"]] = r
    heldout = {}
    with (ds_dir / "heldout_labels.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            heldout[r["sample_id"]] = r

    vec_clusters = {f: set(a["vectors_per_cluster"]) for f, a in folds.items()}
    n_steps = {sid: len(r["steps"]) for sid, r in records.items()}
    selected, pool_sizes = select_stayers(heldout, vec_clusters, n_steps,
                                          args.n_per_cluster, args.seed, args.stayer_rule)
    print(f"[{ds}] stayer pool (held-out incorrect traces, rule={args.stayer_rule}): {pool_sizes}")
    print(f"[{ds}] selected: { {c: len(v) for c, v in selected.items()} }")

    # Per-fold, per-layer patch vectors (unit direction * h_ref * alpha).
    def unit_scaled(v, h_ref):
        return {li: (v[layers.index(li)] / v[layers.index(li)].norm()) * h_ref[layers.index(li)]
                for li in patch_layers}
    fold_patch = {}
    for f, a in folds.items():
        h_ref = a["h_ref_norms"]
        fold_patch[f] = {
            "cluster": {c: {li: t.to(dev_t) for li, t in unit_scaled(v, h_ref).items()}
                        for c, v in a["vectors_per_cluster"].items()},
            "global": {li: t.to(dev_t) for li, t in unit_scaled(a["global_vector"], h_ref).items()},
            "h_ref": {li: float(h_ref[layers.index(li)]) for li in patch_layers},
            "mean": a["mean"].to(dev_b),
            "std": torch.where(a["std"] > 1e-8, a["std"], torch.ones_like(a["std"])).to(dev_b),
            "centers": a["centers"].to(dev_b),
        }

    tag = f"_{args.stayer_rule}_a{args.alpha}" + (f"_{args.tag}" if args.tag else "")
    out_path = ds_dir / f"hit_rate_results{tag}_seed{args.seed}.jsonl"
    summary_path = ds_dir / f"hit_rate_summary{tag}_seed{args.seed}.json"
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                r = json.loads(line)
                done.add((r["sample_id"], r["step_idx"], r["condition"]))
        print(f"[{ds}] resuming: {len(done)} rows already written")

    print(f"[{ds}] loading {args.target_model} on {dev_t}, {BASE_MODEL} on {dev_b}")
    th_tok = AutoTokenizer.from_pretrained(args.target_model)
    th_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev_t).eval()
    prefix_tok = ThinkOpenTemplate(th_tok)
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev_b).eval()
    hidden = th_model.config.hidden_size
    base_backbone = BackboneOnly(base_model)

    patch_state = {"vec_by_layer": None}

    def make_hook(layer_idx):
        # every_token mode of run_eval_meanpool: add to the last position of
        # every forward pass (prefill's last token, then each decoded token).
        def hook(_m, _inp, out):
            vbl = patch_state["vec_by_layer"]
            if vbl is None or layer_idx not in vbl:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h_mod = h.clone()
            h_mod[:, -1, :] = h_mod[:, -1, :] + vbl[layer_idx].to(h.dtype)
            return (h_mod, *out[1:]) if isinstance(out, tuple) else h_mod
        return hook

    handles = [th_model.model.layers[li].register_forward_hook(make_hook(li))
               for li in patch_layers]
    eos = th_tok.eos_token_id
    t0 = time.time()
    out_f = out_path.open("a")
    try:
        for c in sorted(selected):
            for sid, step_idx in tqdm(selected[c], desc=f"{ds} C{c}"):
                if all((sid, step_idx, cond) in done for cond in CONDITIONS):
                    continue
                rec = records[sid]
                fold = heldout[sid]["fold"]
                fp = fold_patch[fold]
                steps_full = rec["steps"]
                question = rec["question"]
                if step_idx + 1 >= len(steps_full):
                    continue
                text_steps = [strip_step_prefix(s) for s in steps_full]
                try:
                    prefix, char_pos = build_instruct_prefix(prefix_tok, question, text_steps, step_idx)
                except ValueError:
                    continue
                enc = th_tok(prefix, return_tensors="pt", return_offsets_mapping=True,
                             add_special_tokens=False)
                offsets = enc["offset_mapping"][0].tolist()
                input_ids = enc["input_ids"].to(dev_t)
                k = first_token_at_or_after(offsets, char_pos)
                if k is None or k >= input_ids.shape[1]:
                    continue
                rng_seed = zlib.crc32(f"{c}|{ds}|{sid}|{step_idx}".encode())

                for cond in CONDITIONS:
                    if (sid, step_idx, cond) in done:
                        continue
                    if cond == "none":
                        vbl = None
                    elif cond in ("steered", "negative"):
                        sign = 1.0 if cond == "steered" else -1.0
                        vbl = {li: sign * args.alpha * v for li, v in fp["cluster"][c].items()}
                    elif cond == "global":
                        vbl = {li: args.alpha * v for li, v in fp["global"].items()}
                    else:  # random, norm-matched to the steered offset
                        g = torch.Generator(device=dev_t).manual_seed(rng_seed)
                        vbl = {}
                        for li in patch_layers:
                            r = torch.randn(hidden, generator=g, device=dev_t)
                            vbl[li] = args.alpha * r * (fp["h_ref"][li] / r.norm())

                    patch_state["vec_by_layer"] = vbl
                    regen_text, n_new, no_marker = regen_one(
                        th_model, th_tok, input_ids, eos, args.max_new_tokens)
                    patch_state["vec_by_layer"] = None

                    if no_marker or not regen_text:
                        regen_cluster, crossed = None, None
                    else:
                        regen_cluster = cluster_step_meanpool(
                            base_backbone, base_tok, fp["mean"], fp["std"], fp["centers"],
                            question, [strip_step_prefix(s) for s in steps_full[: step_idx + 1]],
                            regen_text, dev_b)
                        crossed = regen_cluster != c
                    out_f.write(json.dumps({
                        "dataset": ds, "cluster": c, "fold": fold,
                        "sample_id": int(sid), "step_idx": int(step_idx),
                        "condition": cond, "alpha": args.alpha,
                        "stayer_rule": args.stayer_rule,
                        "regen_cluster": regen_cluster, "crossed": crossed,
                        "no_marker": no_marker, "n_new_tokens": int(n_new),
                        "regen_text": regen_text,
                    }) + "\n")
                    out_f.flush()
                    torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()
        out_f.close()

    # Summary over everything in the file (including resumed rows).
    counts = defaultdict(lambda: {"hit": 0, "n_eval": 0, "no_marker": 0})
    with out_path.open() as f:
        for line in f:
            r = json.loads(line)
            for key in ((r["cluster"], r["condition"]), ("all", r["condition"])):
                if r["crossed"] is None:
                    counts[key]["no_marker"] += 1
                else:
                    counts[key]["n_eval"] += 1
                    counts[key]["hit"] += int(r["crossed"])
    summary = {"dataset": ds, "alpha": args.alpha, "stayer_rule": args.stayer_rule,
               "seed": args.seed, "pool_sizes": pool_sizes,
               "elapsed_sec": time.time() - t0, "by_cluster_condition": {}}
    print(f"\n[{ds}] {'cluster':<8}" + "".join(f"{c:>10}" for c in CONDITIONS))
    for cl in sorted({k[0] for k in counts}, key=str):
        row = []
        for cond in CONDITIONS:
            cc = counts[(cl, cond)]
            rate = cc["hit"] / cc["n_eval"] if cc["n_eval"] else float("nan")
            summary["by_cluster_condition"][f"C{cl}/{cond}"] = {**cc, "rate": rate}
            row.append(f"{rate:>9.1%} ")
        print(f"[{ds}] C{str(cl):<7}" + "".join(row))
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{ds}] results -> {out_path}\n[{ds}] summary -> {summary_path}")


if __name__ == "__main__":
    main()
