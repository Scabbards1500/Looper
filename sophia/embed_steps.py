"""Mean-pool step features for clustering (embedding model: Qwen3-4B-Base).

Per record: one forward pass over question + "\n" + "\n".join(steps); the
last-layer hidden states are averaged over each (prompt, step_1, ..., step_n)
token span. Spans are computed by tokenizing each segment separately and
accumulating lengths, the same rule eval_core.cluster_step_meanpool uses when
re-clustering a regenerated step.

Output: <out-root>/<ds>/embeddings/<sid>.pt, shape [n_steps + 1, hidden]
float32 (row 0 is the prompt). Existing files are skipped, so runs resume.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_full_token_embeddings(text: str, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    return inputs["input_ids"][0].cpu(), out.hidden_states[-1][0]


def locate_token_spans(prompt: str, steps: list[str], tokenizer):
    spans = []
    start = 0
    for seg in [prompt] + steps:
        n = len(tokenizer(seg, add_special_tokens=False)["input_ids"])
        spans.append((start, start + n))
        start += n
    return spans


def build_stepwise_embeddings(token_embeddings, token_spans):
    embeddings = []
    for start, end in token_spans:
        if end <= start:
            if embeddings:
                embeddings.append(embeddings[-1].clone())
            else:
                embeddings.append(torch.zeros(token_embeddings.shape[1],
                                              dtype=token_embeddings.dtype,
                                              device=token_embeddings.device))
            continue
        embeddings.append(token_embeddings[start:end].mean(dim=0))
    return torch.stack(embeddings)


def process_dataset(input_path: Path, out_dir: Path, tokenizer, model, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    n_skipped_nonfinite = 0
    n_skipped_no_steps = 0

    with input_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    for rec in tqdm(records, desc=f"embed {input_path.stem}"):
        sid = rec["sample_id"]
        out_path = out_dir / f"{sid}.pt"
        if out_path.exists():
            n_done += 1
            continue
        prompt = rec.get("question", "")
        steps = [s for s in rec.get("steps", []) if isinstance(s, str) and s.strip()]
        if not steps:
            n_skipped_no_steps += 1
            continue
        full_text = prompt + "\n" + "\n".join(steps)

        _, embs = get_full_token_embeddings(full_text, tokenizer, model, device)
        spans = locate_token_spans(prompt, steps, tokenizer)
        embs = embs.to(torch.float32)

        seq = build_stepwise_embeddings(embs, spans).cpu().contiguous()
        if not torch.isfinite(seq).all():
            n_skipped_nonfinite += 1
            continue

        torch.save(seq, out_path)
        n_done += 1

    return {
        "done": n_done,
        "skipped_nonfinite": n_skipped_nonfinite,
        "skipped_no_steps": n_skipped_no_steps,
        "total": len(records),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="aqua,gsm8k,logiqa,math")
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="directory with <ds>_segmented.jsonl")
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"loading {args.model} on {args.device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        device_map={"": args.device}, attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device

    summary = {}
    for ds in datasets:
        in_path = args.input_dir / f"{ds}_segmented.jsonl"
        out_dir = args.out_root / ds / "embeddings"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        stats = process_dataset(in_path, out_dir, tokenizer, model, device)
        summary[ds] = stats
        print(f"  {ds}: {stats}")

    print("\nsummary:")
    for ds, s in summary.items():
        print(f"  {ds}: done={s['done']}/{s['total']} "
              f"skipped(nonfinite={s['skipped_nonfinite']}, "
              f"no_steps={s['skipped_no_steps']})")


if __name__ == "__main__":
    main()
