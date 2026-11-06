"""Cache steering-target residual states at the start of every step.

Input text is question + "\n" + "\n".join(steps) with no chat template. For
each step the state is read at the first token at or after its discourse
marker (i.e. after the "Step N: " prefix), at every patched layer. Intro
steps and the last step of each trace are skipped, as are records whose
state_labels length differs from steps. Output: <out-dir>/<ds>.pt with
states [N, n_layers, hidden] (float32), layers, dataset, sample_id, step_idx,
phase ("all"), is_crosser, model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sophia.eval_core import first_token_at_or_after


def parse_layers(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layers", required=True, help="e.g. 18-35")
    p.add_argument("--clustered-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--datasets", default="gsm8k,aqua,logiqa,math")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    layers = parse_layers(args.layers)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa").to(args.device).eval()
    captured: dict[int, torch.Tensor] = {}

    def make_hook(li):
        def hook(_m, _inp, out):
            captured[li] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in layers]
    try:
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            out_path = args.out_dir / f"{ds}.pt"
            if out_path.exists() and args.limit is None:
                print(f"[{ds}] exists, skipping")
                continue
            lines = (args.clustered_dir / f"{ds}_clustered.jsonl").read_text().splitlines()
            if args.limit is not None:
                lines = lines[: args.limit]
            states, meta = [], {"dataset": [], "sample_id": [], "step_idx": [], "phase": [], "is_crosser": []}
            for line in tqdm(lines, desc=ds):
                rec = json.loads(line)
                steps, labels = rec["steps"], rec["state_labels"]
                step_labels = rec.get("step_labels", [])
                if len(steps) < 2 or len(labels) != len(steps):
                    continue
                wanted = [(i, labels[i] != labels[i + 1]) for i in range(len(steps) - 1)
                          if not (i < len(step_labels) and step_labels[i] == "intro")]
                if not wanted:
                    continue
                full_text = rec["question"] + "\n" + "\n".join(steps)
                enc = tok(full_text, return_tensors="pt", return_offsets_mapping=True,
                          add_special_tokens=False)
                offsets = enc["offset_mapping"][0].tolist()
                with torch.no_grad():
                    # Backbone only: the hooked decoder-layer outputs are the
                    # same computation, minus full-sequence LM-head logits
                    # (~5 GB for a 16k-token 8B trace).
                    model.model(input_ids=enc["input_ids"].to(args.device))
                # Copy to CPU before the (exact) bf16->fp32 cast: for a 30k-token
                # 8B trace the GPU-side fp32 stack alone is ~8 GB.
                hidden = torch.stack([captured[li][0].cpu() for li in layers], 0).to(torch.float32)
                char_pos = len(rec["question"]) + 1
                marker_chars = []
                for step in steps:
                    colon = step.find(": ")
                    marker_chars.append(char_pos + (colon + 2 if colon >= 0 else 0))
                    char_pos += len(step) + 1
                T = hidden.shape[1]
                for i, crosser in wanted:
                    ti = first_token_at_or_after(offsets, marker_chars[i])
                    if ti is None or ti >= T:
                        continue
                    states.append(hidden[:, ti, :].clone())
                    meta["dataset"].append(ds)
                    meta["sample_id"].append(rec["sample_id"])
                    meta["step_idx"].append(i)
                    meta["phase"].append("all")
                    meta["is_crosser"].append(bool(crosser))
                del enc, hidden
                captured.clear()
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            st = torch.stack(states)
            torch.save({"states": st, "layers": layers, **meta, "model": args.model, "datasets": [ds]},
                       out_path)
            print(f"[{ds}] saved {tuple(st.shape)} -> {out_path}")
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
