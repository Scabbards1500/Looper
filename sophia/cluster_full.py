"""Full-data K-means over mean-pool step features (the reference partition).

Feature rows: row i+1 of <sid>.pt is step i; non-finite rows are dropped.
Features are z-scored over all steps, then KMeans(n_clusters=5,
random_state=42, n_init=10). Cluster ids are arbitrary and model-local.

Writes <out-dir>/<ds>_clustered.jsonl (segmented records + state_labels) and
<artifact-dir>/<ds>_kmeans.pt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segmented-dir", required=True, type=Path)
    p.add_argument("--embed-root", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--artifact-dir", required=True, type=Path)
    p.add_argument("--datasets", default="gsm8k,aqua,logiqa,math")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-init", type=int, default=10)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        records = [json.loads(l) for l in (args.segmented_dir / f"{ds}_segmented.jsonl").open()]
        X, owners = [], []
        for rec in records:
            ep = args.embed_root / ds / "embeddings" / f"{rec['sample_id']}.pt"
            if not ep.exists():
                continue
            emb = torch.load(ep, weights_only=False)
            for i in range(len(rec["steps"])):
                if i + 1 >= emb.shape[0]:
                    continue
                v = emb[i + 1].float().numpy()
                if np.isfinite(v).all():
                    X.append(v)
                    owners.append((rec["sample_id"], i))
        X = np.stack(X)
        mean, std = X.mean(0), X.std(0)
        Z = (X - mean) / np.where(std > 1e-8, std, 1.0)
        km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=args.n_init).fit(Z)
        by_sid: dict[int, list] = {}
        for (sid, i), c in zip(owners, km.labels_):
            by_sid.setdefault(sid, []).append((i, int(c)))
        n_with = 0
        with (args.out_dir / f"{ds}_clustered.jsonl").open("w") as f:
            for rec in records:
                lab = by_sid.get(rec["sample_id"])
                if lab is not None:
                    rec["state_labels"] = [c for _, c in sorted(lab)]
                    n_with += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        torch.save({"centers": torch.from_numpy(km.cluster_centers_).float(),
                    "mean": torch.from_numpy(mean).float(), "std": torch.from_numpy(std).float(),
                    "match_rate": 0.0, "match_rate_aligned": False, "n_steps": int(X.shape[0]),
                    "n_clusters": args.k, "seed": args.seed, "n_init": args.n_init},
                   args.artifact_dir / f"{ds}_kmeans.pt")
        print(f"[{ds}] steps={X.shape[0]} traces_labelled={n_with}/{len(records)} "
              f"sizes={np.bincount(km.labels_, minlength=args.k).tolist()}")


if __name__ == "__main__":
    main()
