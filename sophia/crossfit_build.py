"""Cross-fitted (trace-level held-out) offline stages of SOPHIA.

Every trace is assigned to one of F folds (stratified by is_correct). For
fold f, all offline statistics are fit on the other F-1 folds only:

  - z-score (mean, std) and KMeans(K=5, random_state=42, n_init=--n-init) on
    the mean-pool step features of training traces;
  - fold cluster ids aligned to the full-data reference partition
    (cluster_full.py) by Hungarian matching on training steps, so "C2" means
    the same thing in every fold;
  - held-out traces labelled by nearest fold centroid;
  - per-cluster crosser-minus-stayer vectors from cached residual rows of
    training traces only, using the fold's labels;
  - one pooled "global" crosser-minus-stayer vector (all clusters together),
    the state-agnostic baseline.

Every eval prefix is then drawn from a trace held out of its fold's fit.

Output per dataset (<out_root>/<ds>/):
  fold{f}.pt            fold artifact consumed by crossfit_eval.py
  heldout_labels.jsonl  {sample_id, fold, state_labels (held-out), is_correct}
  build_summary.json    agreement with the full-data partition and cosine
                        between fold and full-data vectors (stability checks)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

K = 5
KMEANS_SEED = 42


def load_records(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def collect_features(records, embed_dir: Path):
    """Same row convention as cluster_meanpool.collect_features: row i+1 of
    <sid>.pt is step i; rows with non-finite values are dropped."""
    X, y_ref, owners = [], [], []
    for rec in records:
        sid = rec["sample_id"]
        p = embed_dir / f"{sid}.pt"
        labels = rec.get("state_labels")
        if not p.exists() or labels is None or len(labels) != len(rec["steps"]):
            continue
        emb = torch.load(p, weights_only=False)
        for i in range(len(rec["steps"])):
            if i + 1 >= emb.shape[0]:
                continue
            v = emb[i + 1].float().numpy()
            if not np.isfinite(v).all():
                continue
            X.append(v)
            y_ref.append(labels[i])
            owners.append((sid, i))
    return np.stack(X), np.array(y_ref), owners


def assign_folds(records, n_folds: int, seed: int) -> dict[int, int]:
    """Stratified by is_correct so the scarce incorrect traces (e.g. 24 on
    gsm8k) spread evenly across folds."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for flag in (True, False):
        sids = sorted(r["sample_id"] for r in records if bool(r.get("is_correct")) == flag)
        rng.shuffle(sids)
        for j, sid in enumerate(sids):
            fold_of[sid] = j % n_folds
    return fold_of


def align_to_reference(y_ref: np.ndarray, y_fold: np.ndarray, k: int):
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(y_ref, y_fold):
        cm[a, b] += 1
    r, c = linear_sum_assignment(-cm)
    perm = {int(cc): int(rr) for rr, cc in zip(r, c)}
    aligned = np.array([perm[x] for x in y_fold])
    return perm, float((aligned == y_ref).mean())


def group_means(states: torch.Tensor, group: np.ndarray, n_groups: int):
    """Mean of states rows per group id (rows with group < 0 ignored).
    Returns sums-based means [n_groups, L, D] and counts [n_groups]."""
    keep = group >= 0
    idx = torch.from_numpy(group[keep]).long()
    rows = states[torch.from_numpy(keep)]
    sums = torch.zeros((n_groups,) + tuple(states.shape[1:]), dtype=torch.float64)
    sums.index_add_(0, idx, rows.double())
    counts = torch.bincount(idx, minlength=n_groups)
    means = sums / counts.clamp(min=1).view(-1, 1, 1).double()
    return means.float(), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="gsm8k,aqua,logiqa,math")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=0)
    # The full-data reference uses n_init=10; inside folds 10 restarts can
    # land in a worse local optimum (Qwen3-4B logiqa fold 3: inertia 6.434e7
    # vs 6.419e7), which 50 restarts avoid.
    ap.add_argument("--n-init", type=int, default=50)
    ap.add_argument("--clustered-dir", type=Path, required=True,
                    help="full-data partition from cluster_full.py")
    ap.add_argument("--embed-root", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="residual cache from extract_states.py")
    ap.add_argument("--full-vectors", type=Path, required=True,
                    help="full-data vectors from full_vectors.py (cosine reference)")
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        out_dir = args.out_root / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        records = load_records(args.clustered_dir / f"{ds}_clustered.jsonl")
        rec_by_sid = {r["sample_id"]: r for r in records}
        print(f"\n=== {ds}: {len(records)} traces, "
              f"{sum(not r.get('is_correct') for r in records)} incorrect")

        X, y_ref, owners = collect_features(records, args.embed_root / ds / "embeddings")
        owner_sid = np.array([s for s, _ in owners])
        fold_of = assign_folds(records, args.folds, args.fold_seed)
        step_fold = np.array([fold_of[s] for s in owner_sid])
        print(f"  features {X.shape}")

        cache = torch.load(args.cache_dir / f"{ds}.pt", map_location="cpu", weights_only=False)
        states = cache["states"]
        layers = list(cache["layers"])
        c_sid = np.array(cache["sample_id"])
        c_step = np.array(cache["step_idx"])

        full_sv = torch.load(args.full_vectors / f"steering_vectors_{ds}.pt",
                              map_location="cpu", weights_only=False)

        heldout_labels: dict[int, list[int]] = {}
        summary = {"dataset": ds, "folds": args.folds, "fold_seed": args.fold_seed,
                   "kmeans_n_init": args.n_init,
                   "n_steps": int(X.shape[0]), "per_fold": []}

        for f in range(args.folds):
            tr = step_fold != f
            mean = X[tr].mean(axis=0)
            std = X[tr].std(axis=0)
            std_safe = np.where(std > 1e-8, std, 1.0)
            Z = (X - mean) / std_safe
            km = KMeans(n_clusters=K, random_state=KMEANS_SEED, n_init=args.n_init).fit(Z[tr])
            y_raw = km.predict(Z)
            perm, match_train = align_to_reference(y_ref[tr], y_raw[tr], K)
            y_fold = np.array([perm[c] for c in y_raw])
            centers = np.zeros_like(km.cluster_centers_)
            for raw_id, ref_id in perm.items():
                centers[ref_id] = km.cluster_centers_[raw_id]
            agree_heldout = float((y_fold[~tr] == y_ref[~tr]).mean())

            # Per-trace fold labels (train traces: used for vectors; held-out
            # traces: used for eval-prefix selection).
            labels_by_sid: dict[int, dict[int, int]] = {}
            for (sid, i), c in zip(owners, y_fold):
                labels_by_sid.setdefault(sid, {})[i] = int(c)
            for sid, d in labels_by_sid.items():
                if fold_of[sid] == f:
                    n = len(rec_by_sid[sid]["steps"])
                    heldout_labels[sid] = [d.get(i, -1) for i in range(n)]

            # Group cached residual rows of TRAIN traces: g = 2*c + is_crosser.
            group = np.full(len(c_sid), -1, dtype=np.int64)
            for r, (sid, i) in enumerate(zip(c_sid, c_step)):
                if fold_of.get(int(sid), f) == f:
                    continue
                d = labels_by_sid.get(int(sid))
                if d is None or i not in d or (i + 1) not in d:
                    continue
                cur, nxt = d[i], d[i + 1]
                group[r] = 2 * cur + int(cur != nxt)
            means, counts = group_means(states, group, 2 * K)

            vectors, stayer_means, cnt = {}, {}, {}
            for c in range(K):
                ns, nc = int(counts[2 * c]), int(counts[2 * c + 1])
                cnt[c] = {"c": nc, "s": ns}
                if nc == 0 or ns == 0:
                    continue
                vectors[c] = means[2 * c + 1] - means[2 * c]
                stayer_means[c] = means[2 * c]

            # Pooled, state-agnostic crosser-minus-stayer direction.
            pooled = np.where(group >= 0, group % 2, -1)
            pmeans, pcounts = group_means(states, pooled, 2)
            global_vec = pmeans[1] - pmeans[0]

            # h_ref per layer: average stayer-mean norm across clusters, same
            # as run_eval_meanpool.py --normalize.
            h_ref = torch.stack([m.norm(dim=-1) for m in stayer_means.values()]).mean(0)

            cos_full = {}
            for c, v in vectors.items():
                if c in full_sv["vectors_per_cluster"]:
                    pv = full_sv["vectors_per_cluster"][c]
                    cos_full[c] = float(torch.nn.functional.cosine_similarity(v, pv, dim=-1).mean())
            cos_global = {c: float(torch.nn.functional.cosine_similarity(v, global_vec, dim=-1).mean())
                          for c, v in vectors.items()}

            torch.save({
                "dataset": ds, "fold": f, "layers": layers, "kmeans_n_init": args.n_init,
                "mean": torch.from_numpy(mean).float(),
                "std": torch.from_numpy(std).float(),
                "centers": torch.from_numpy(centers).float(),
                "vectors_per_cluster": vectors,
                "global_vector": global_vec,
                "h_ref_norms": h_ref,
                "counts": cnt,
                "global_counts": {"c": int(pcounts[1]), "s": int(pcounts[0])},
                "train_sids": sorted(s for s, ff in fold_of.items() if ff != f),
                "heldout_sids": sorted(s for s, ff in fold_of.items() if ff == f),
            }, out_dir / f"fold{f}.pt")

            info = {"fold": f, "match_rate_train_vs_full": match_train,
                    "heldout_agreement_vs_full": agree_heldout,
                    "counts": cnt, "global_counts": {"c": int(pcounts[1]), "s": int(pcounts[0])},
                    "mean_cos_vs_full_vector": cos_full,
                    "mean_cos_cluster_vs_global": cos_global}
            summary["per_fold"].append(info)
            print(f"  fold {f}: match(train)={match_train:.3f} "
                  f"agree(held-out)={agree_heldout:.3f} "
                  f"vectors={sorted(vectors)} "
                  f"cos(full)={ {c: round(v, 3) for c, v in cos_full.items()} } "
                  f"cos(global)={ {c: round(v, 3) for c, v in cos_global.items()} }")

        with (out_dir / "heldout_labels.jsonl").open("w") as fo:
            for r in records:
                sid = r["sample_id"]
                if sid in heldout_labels:
                    fo.write(json.dumps({"sample_id": sid, "fold": fold_of[sid],
                                         "is_correct": bool(r.get("is_correct")),
                                         "state_labels": heldout_labels[sid]}) + "\n")
        with (out_dir / "build_summary.json").open("w") as fo:
            json.dump(summary, fo, indent=2)
        print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
