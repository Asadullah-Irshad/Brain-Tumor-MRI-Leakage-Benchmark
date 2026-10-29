"""Step 2 - repeated, leakage-aware cross-validation on the TRAINING partition

protocol 'grouped' : StratifiedGroupKFold - no duplicate group / patient crosses folds
protocol 'naive'   : StratifiedKFold at image level (reproduces the original
                     submission's protocol; used ONLY to measure leakage inflation)

Every run uses a group-aware validation split taken from the training folds
for early stopping, so the test fold never influences training or model choice.
Runs are resumable: a finished (model, mode, seed, fold) is skipped.

usage:
  python scripts/s02_cv.py --data_root DATA --groups results/groups.csv --out results \
      --models efficientnet_b0 --modes two_stage --seeds 0 1 2 --protocol grouped
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import (Config, build_cache, build_model, compute_metrics, count_params, fit,
                    group_val_split, index_nickparvar, predict, save_json, seed_everything,
                    environment_info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["efficientnet_b0"])
    ap.add_argument("--modes", nargs="+", default=["two_stage"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--protocol", choices=["grouped", "naive"], default="grouped")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    cfg = Config()
    if a.smoke:
        cfg.smoke()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    df = index_nickparvar(a.data_root)
    g = pd.read_csv(a.groups)[["rel", "group_id"]]
    df = df.merge(g, on="rel", how="left")
    assert df.group_id.notna().all(), "groups.csv does not match data_root - rerun s01_dedup.py"
    tr = df[df.split == "train"].reset_index(drop=True)
    imgs = build_cache(tr["path"].tolist(), cfg.img_size, os.path.join(a.cache_dir, "train"))
    y, groups = tr["label"].values, tr["group_id"].values

    out_dir = os.path.join(a.out, "cv", a.protocol)
    os.makedirs(out_dir, exist_ok=True)
    save_json({"config": cfg.__dict__, "env": environment_info()}, os.path.join(out_dir, "run_info.json"))

    for seed in a.seeds:
        if a.protocol == "grouped":
            splitter = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(len(y)), y, groups))
        else:
            splitter = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(len(y)), y))
        for fold, (tr_idx, te_idx) in enumerate(splits):
            # group-aware validation split inside the training folds
            itr, iva = group_val_split(groups[tr_idx], y[tr_idx], cfg.val_fraction, seed)
            fit_idx, val_idx = tr_idx[itr], tr_idx[iva]
            for model_name in a.models:
                for mode in a.modes:
                    tag = f"{model_name}__{mode}__s{seed}_f{fold}"
                    fjson = os.path.join(out_dir, tag + ".json")
                    if os.path.exists(fjson):
                        print("skip (done):", tag)
                        continue
                    print(f"=== {a.protocol} | {tag} | fit {len(fit_idx)} val {len(val_idx)} test {len(te_idx)}")
                    seed_everything(seed)
                    model, head = build_model(model_name, cfg)
                    t0 = time.time()
                    hist = fit(model, head, imgs[fit_idx], y[fit_idx], imgs[val_idx], y[val_idx],
                               mode, cfg, dev)
                    p = predict(model, imgs[te_idx], cfg, dev)
                    m = compute_metrics(y[te_idx], p)
                    m.update({"model": model_name, "mode": mode, "seed": seed, "fold": fold,
                              "protocol": a.protocol, "n_fit": len(fit_idx), "n_val": len(val_idx),
                              "n_test": len(te_idx), "params": count_params(model),
                              "train_minutes": (time.time() - t0) / 60, "history": hist,
                              "group_overlap_with_test": int(len(set(groups[fit_idx]) & set(groups[te_idx])))})
                    np.save(os.path.join(out_dir, tag + "_probs.npy"), p)
                    np.save(os.path.join(out_dir, tag + "_idx.npy"), te_idx)
                    save_json(m, fjson)
                    print(f"    acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} "
                          f"AUC {m['macro_auc']:.4f} ({m['train_minutes']:.1f} min)")
                    del model
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
    tr[["rel", "cls", "label", "group_id"]].to_csv(os.path.join(out_dir, "train_index.csv"), index=False)


if __name__ == "__main__":
    main()
