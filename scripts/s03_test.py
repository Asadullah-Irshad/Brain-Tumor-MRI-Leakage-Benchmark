"""Step 3 - final models and held-out / external testing

For each (model, mode, seed) a final model is trained on the whole training
partition (a group-aware 15% of it is used only for early stopping) and then
evaluated ONCE on:
  official : the dataset's own Testing folder (same source; contains images
             with near-duplicates in Training - reported for comparability only)
  clean    : official test images with NO duplicate/patient twin in Training
  external : an independently collected dataset (optional, --external_dir)

The test sets never influence training, early stopping or model choice.

usage:
  python scripts/s03_test.py --data_root DATA --groups results/groups.csv --out results \
      --models efficientnet_b0 --modes two_stage --seeds 0 1 2 [--external_dir EXT]
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import (CLASSES, Config, bootstrap_ci, build_cache, build_model, compute_metrics,
                    count_params, fit, group_val_split, index_external, index_nickparvar,
                    predict, save_json, seed_everything, environment_info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["efficientnet_b0"])
    ap.add_argument("--modes", nargs="+", default=["two_stage"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--external_dir", default=None)
    ap.add_argument("--external_name", default="external")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    cfg = Config()
    if a.smoke:
        cfg.smoke()
        a.n_boot = 50
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = index_nickparvar(a.data_root)
    g = pd.read_csv(a.groups)[["rel", "group_id", "test_has_train_twin"]]
    df = df.merge(g, on="rel", how="left")
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    tr_imgs = build_cache(tr["path"].tolist(), cfg.img_size, os.path.join(a.cache_dir, "train"))
    te_imgs = build_cache(te["path"].tolist(), cfg.img_size, os.path.join(a.cache_dir, "test"))
    y_tr, g_tr, y_te = tr["label"].values, tr["group_id"].values, te["label"].values
    clean_mask = ~te["test_has_train_twin"].astype(bool).values

    sets = {"official": (te_imgs, y_te, te["rel"].values),
            "clean": (te_imgs[np.where(clean_mask)[0]], y_te[clean_mask], te["rel"].values[clean_mask])}
    if a.external_dir:
        ex = index_external(a.external_dir).reset_index(drop=True)
        ex_imgs = build_cache(ex["path"].tolist(), cfg.img_size, os.path.join(a.cache_dir, "external"))
        ovl = os.path.join(a.out, "external_overlap.csv")
        if os.path.exists(ovl):  # drop external images that duplicate the main dataset
            o = pd.read_csv(ovl)[["rel", "has_twin_in_main_dataset"]]
            ex = ex.merge(o, on="rel", how="left").fillna({"has_twin_in_main_dataset": False})
            keep = ~ex["has_twin_in_main_dataset"].astype(bool).values
            print(f"external: dropping {int((~keep).sum())} images that duplicate the main dataset")
        else:
            keep = np.ones(len(ex), bool)
        sets["external"] = (ex_imgs[np.where(keep)[0]], ex["label"].values[keep], ex["rel"].values[keep])

    out_dir = os.path.join(a.out, "test")
    os.makedirs(out_dir, exist_ok=True)
    save_json({"config": cfg.__dict__, "env": environment_info(),
               "external_name": a.external_name if a.external_dir else None},
              os.path.join(out_dir, "run_info.json"))

    for seed in a.seeds:
        itr, iva = group_val_split(g_tr, y_tr, cfg.val_fraction, seed)
        for model_name in a.models:
            for mode in a.modes:
                tag = f"{model_name}__{mode}__s{seed}"
                fjson = os.path.join(out_dir, tag + ".json")
                if os.path.exists(fjson):
                    print("skip (done):", tag)
                    continue
                print(f"=== final model {tag} | fit {len(itr)} val {len(iva)}")
                seed_everything(seed)
                model, head = build_model(model_name, cfg)
                t0 = time.time()
                hist = fit(model, head, tr_imgs[itr], y_tr[itr], tr_imgs[iva], y_tr[iva], mode, cfg, dev)
                res = {"model": model_name, "mode": mode, "seed": seed, "params": count_params(model),
                       "train_minutes": (time.time() - t0) / 60, "history": hist, "sets": {}}
                for sname, (imgs, yy, rels) in sets.items():
                    if len(yy) == 0:
                        continue
                    p = predict(model, imgs, cfg, dev)
                    m = compute_metrics(yy, p)
                    m["bootstrap_ci_95"] = bootstrap_ci(yy, p, a.n_boot, seed=seed)
                    res["sets"][sname] = m
                    np.save(os.path.join(out_dir, f"{tag}__{sname}_probs.npy"), p)
                    pd.DataFrame({"rel": rels, "true": [CLASSES[k] for k in yy],
                                  "pred": [CLASSES[k] for k in p.argmax(1)],
                                  "confidence": p.max(1), **{f"p_{c}": p[:, i] for i, c in enumerate(CLASSES)}}
                                 ).to_csv(os.path.join(out_dir, f"{tag}__{sname}_predictions.csv"), index=False)
                    print(f"    {sname:9s} n={len(yy):5d} acc {m['accuracy']:.4f} F1 {m['macro_f1']:.4f} "
                          f"AUC {m['macro_auc']:.4f} glioma recall {m['per_class']['glioma']['recall']:.3f}")
                torch.save(model.state_dict(), os.path.join(out_dir, tag + ".pt"))
                save_json(res, fjson)
                del model
                if dev.type == "cuda":
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
