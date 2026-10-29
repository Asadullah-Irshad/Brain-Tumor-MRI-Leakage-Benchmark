"""Step 0 - dataset audit.

Counts every image per class in the Training and Testing folders exactly as
distributed, records basic file properties, and states which public release
of the dataset the counts correspond to.

usage: python scripts/s00_audit.py --data_root /path/to/brain-tumor-mri-dataset --out results
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from common import CLASSES, index_nickparvar, save_json

KNOWN_RELEASES = {
    "v1 (7,023 images)": {"train": [1321, 1339, 1595, 1457], "test": [300, 306, 405, 300]},
    "v2 (7,200 images)": {"train": [1400, 1400, 1400, 1400], "test": [400, 400, 400, 400]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = index_nickparvar(a.data_root)
    print(f"indexed {len(df)} images")
    props = []
    for p in df["path"]:
        with Image.open(p) as im:
            w, h = im.size
            props.append({"width": w, "height": h, "mode": im.mode,
                          "format": (im.format or "").upper(), "bytes": os.path.getsize(p)})
    df = pd.concat([df, pd.DataFrame(props)], axis=1)
    os.makedirs(a.out, exist_ok=True)
    df.drop(columns=["path"]).to_csv(os.path.join(a.out, "audit_index.csv"), index=False)

    counts = df.pivot_table(index="cls", columns="split", values="rel", aggfunc="count").reindex(CLASSES).fillna(0).astype(int)
    print(counts)
    tr = [int(counts.loc[c, "train"]) for c in CLASSES]
    te = [int(counts.loc[c, "test"]) for c in CLASSES]
    release = "unrecognised (check the Kaggle version you downloaded)"
    for name, ref in KNOWN_RELEASES.items():
        if ref["train"] == tr and ref["test"] == te:
            release = name

    # exact byte-identical files (quick check; s01 does the thorough check)
    summary = {
        "release_detected": release,
        "train_per_class": dict(zip(CLASSES, tr)), "test_per_class": dict(zip(CLASSES, te)),
        "n_train": int(sum(tr)), "n_test": int(sum(te)), "n_total": int(sum(tr) + sum(te)),
        "size_summary": {
            s: {c: {"median_w": float(g["width"].median()), "median_h": float(g["height"].median()),
                    "pct_grayscale_mode": float((g["mode"] == "L").mean() * 100),
                    "formats": g["format"].value_counts().to_dict()}
                for c, g in df[df.split == s].groupby("cls")}
            for s in ["train", "test"]},
        "oversampling_or_copying_by_author": "none - images are used exactly as distributed",
    }
    save_json(summary, os.path.join(a.out, "audit_summary.json"))
    print("release detected:", release)
    print("wrote", os.path.join(a.out, "audit_summary.json"))


if __name__ == "__main__":
    main()
