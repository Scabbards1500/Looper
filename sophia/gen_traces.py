"""Generate chain-of-thought traces for a Qwen3 steering target with vLLM.

First `--limit` examples per dataset (sophia.tasks), sampling with
temperature 0.6, top_p 0.95, seed 0, up to 16,384 new tokens. Each output row
stores the trace (prefixed with "<think>\n" when the prompt opened the think
block), the first \boxed{} prediction and an exact-match `correct` flag
(build_segmented.py later recomputes correctness as `is_correct`).

Hybrid Qwen3 models (1.7B, 8B) end their generation prompt at
"<|im_start|>assistant\n", while Qwen3-4B-Thinking-2507 ends it with
"<think>\n"; "<think>\n" is appended when missing so every model sees the same
prompt shape.

Runs several datasets in one vLLM instance; set CUDA_VISIBLE_DEVICES to one GPU.
"""
import argparse
import json
import os
import re
from pathlib import Path

from sophia.tasks import load_examples

THINK_OPEN = "<think>\n"


def build_prompt(tok, question: str) -> str:
    prompt = tok.apply_chat_template([{"role": "user", "content": question}],
                                     tokenize=False, add_generation_prompt=True)
    return prompt if prompt.endswith(THINK_OPEN) else prompt + THINK_OPEN


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--datasets", default="gsm8k,aqua,logiqa,math")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--logiqa-path", default=None,
                   help="LogiQA Train.txt (default: $SOPHIA_LOGIQA or data/logiqa_train.txt)")
    args = p.parse_args()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} model={args.model}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_tokens + 1024, seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, seed=args.seed)
    boxed_re = re.compile(r"\\boxed\{([^{}]+)\}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        out_path = args.out_dir / f"{ds}.jsonl"
        if out_path.exists():
            print(f"[{ds}] exists, skipping: {out_path}", flush=True)
            continue
        examples = load_examples(ds, args.limit, args.logiqa_path)
        prompts = [build_prompt(tok, ex["question"]) for ex in examples]
        print(f"[{ds}] {len(examples)} examples; prompt tail: {prompts[0][-60:]!r}", flush=True)
        outputs = llm.generate(prompts, sp)
        tmp = out_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for ex, o in zip(examples, outputs):
                text = o.outputs[0].text
                # Same rule as the 4B script: the prompt opened <think>, so the
                # saved trace is prefixed with it for the segmenter.
                text_for_save = text if "<think>" in text else THINK_OPEN + text
                m = boxed_re.search(text_for_save)
                pred_raw = m.group(1).strip() if m else None
                correct = (pred_raw is not None and
                           pred_raw.replace(" ", "") == str(ex["gold_norm"]).replace(" ", ""))
                f.write(json.dumps({
                    "sample_id": ex["sample_id"], "question": ex["question"],
                    "gold_answer": ex["gold_answer"], "gold_norm": ex["gold_norm"],
                    "generated_responses": [text_for_save], "pred_raw": pred_raw,
                    "correct": bool(correct), "n_generated_tokens": len(o.outputs[0].token_ids),
                }, ensure_ascii=False) + "\n")
        tmp.rename(out_path)
        print(f"[{ds}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
