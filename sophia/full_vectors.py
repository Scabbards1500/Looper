"""Full-data per-cluster crosser-minus-stayer steering vectors.

Groups the cached residual states (extract_states.py) by the full-data
partition (cluster_full.py): a cached step i in cluster c is a crosser when
step i+1 is in a different cluster, otherwise a stayer. For every cluster with
both kinds of rows, vector = crosser mean - stayer mean, at every cached layer.
Used as the reference that per-fold vectors are compared against.

Output (--out) holds vectors_per_cluster {c: [n_layers, D]},
means_per_cluster, counts, layers, model, datasets and the pairwise cosine
between cluster vectors at each layer.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

DEFAULT_DATASETS = ("gsm8k", "aqua", "logiqa", "math")
N_CLUSTERS = 5


def load_new_labels(path: Path) -> dict:
    """sample_id -> list[int] of new state_labels."""
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            out[sid] = rec.get("state_labels", [])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--clustered-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--k", type=int, default=N_CLUSTERS)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    new_labels_per_ds = {}
    for ds in datasets:
        new_path = args.clustered_dir / f"{ds}_clustered.jsonl"
        if not new_path.exists():
            raise SystemExit(f"missing new clustered file for {ds}: {new_path}")
        new_labels_per_ds[ds] = load_new_labels(new_path)

    states_all = []
    cluster_all = []
    crosser_all = []
    layers_ref = None
    model_ref = None
    n_drop_no_next = 0
    n_drop_no_record = 0

    for ds in datasets:
        cache_path = args.cache_dir / f"{ds}.pt"
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        states = d["states"]  # [N, n_layers, D]
        sample_ids = d["sample_id"]
        step_idxs = d["step_idx"]
        layers = list(d["layers"])
        model_name = d["model"]
        if layers_ref is None:
            layers_ref = layers
            model_ref = model_name
        else:
            assert layers == layers_ref, f"layer mismatch on {ds}"
            assert model_name == model_ref, f"model mismatch on {ds}"

        new_labels = new_labels_per_ds[ds]
        clusters_ds = []
        crossers_ds = []
        keep_mask = []
        for i, (sid, step_idx) in enumerate(zip(sample_ids, step_idxs)):
            labels = new_labels.get(sid)
            if labels is None:
                n_drop_no_record += 1
                keep_mask.append(False)
                continue
            if step_idx + 1 >= len(labels) or step_idx >= len(labels):
                n_drop_no_next += 1
                keep_mask.append(False)
                continue
            cur = labels[step_idx]
            nxt = labels[step_idx + 1]
            clusters_ds.append(int(cur))
            crossers_ds.append(bool(cur != nxt))
            keep_mask.append(True)

        keep_mask_t = torch.tensor(keep_mask, dtype=torch.bool)
        kept_states = states[keep_mask_t]
        states_all.append(kept_states)
        cluster_all.extend(clusters_ds)
        crosser_all.extend(crossers_ds)
        print(f"  {ds}: cache {states.shape[0]}, kept {kept_states.shape[0]}")

    states_t = torch.cat(states_all, dim=0)
    cluster_t = torch.tensor(cluster_all, dtype=torch.long)
    crosser_t = torch.tensor(crosser_all, dtype=torch.bool)
    print(f"\ntotal kept: {states_t.shape[0]} / dropped(no_record={n_drop_no_record}, "
          f"no_next={n_drop_no_next})")

    vectors_per_cluster = {}
    means_per_cluster = {}
    counts = {}

    for c in range(args.k):
        c_mask = (cluster_t == c) & crosser_t
        s_mask = (cluster_t == c) & ~crosser_t
        nc, ns = int(c_mask.sum()), int(s_mask.sum())
        counts[c] = {"c": nc, "s": ns}
        if nc == 0 or ns == 0:
            print(f"[warn] cluster {c}: c={nc} s={ns} — skipping vector")
            continue
        c_mean = states_t[c_mask].mean(dim=0)  # [n_layers, D]
        s_mean = states_t[s_mask].mean(dim=0)
        diff = c_mean - s_mean
        vectors_per_cluster[c] = diff
        means_per_cluster[c] = {"crosser_mean": c_mean, "stayer_mean": s_mean}

    # Pairwise cosine across the K crosser-minus-stayer vectors at each layer.
    pairwise = {}
    keys = sorted(vectors_per_cluster.keys())
    for li, layer in enumerate(layers_ref):
        rows = torch.stack([vectors_per_cluster[c][li] for c in keys])  # [K, D]
        rows_n = rows / (rows.norm(dim=-1, keepdim=True) + 1e-12)
        M = rows_n @ rows_n.T
        pairwise[int(layer)] = M

    out_dict = {
        "vectors_per_cluster": vectors_per_cluster,
        "means_per_cluster": means_per_cluster,
        "counts": counts,
        "layers": layers_ref,
        "model": model_ref,
        "datasets": datasets,
        "pairwise_cosine_per_layer": pairwise,
        "cluster_keys": keys,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, args.out)
    print(f"\nsaved to {args.out}")

    print("\nper-cluster crosser-vs-stayer counts:")
    for c, cs in counts.items():
        print(f"  C{c}: c={cs['c']} s={cs['s']}")

    print("\npairwise cosine across cluster steering vectors per layer:")
    print("layer | mean | max | min")
    for layer in layers_ref:
        M = pairwise[int(layer)].numpy()
        n = M.shape[0]
        iu = np.triu_indices(n, k=1)
        vals = M[iu]
        print(f"  {layer:3d}  | {vals.mean():+.3f} | {vals.max():+.3f} | "
              f"{vals.min():+.3f}")


if __name__ == "__main__":
    main()
