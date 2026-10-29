"""Step 1 - duplicate / near-duplicate / patient leakage audit

Method (follows the recent audits of this dataset, Duong et al. 2026 and
Vangala et al. 2026):
  1. exact duplicates   : MD5 of the decoded grey-level pixel array
  2. near-duplicate candidates : 64-bit perceptual hash (DCT pHash), made
     invariant to the 8 rotations/reflections, Hamming distance <= --phash_thr
  3. confirmation       : normalised cross-correlation (NCC) of 64x64
     thumbnails, max over the 8 rotations/reflections, >= --ncc_thr
  4. optional patient recovery : images matched (NCC >= --ncc_thr) to the
     figshare release of Cheng et al., whose .mat files carry a patient ID
  5. union-find over all links -> group_id. Images in one group never cross a
     cross-validation fold, and a test image whose group contains any training
     image is excluded from the 'clean' test set.

usage:
  python scripts/s01_dedup.py --data_root DATA --out results [--figshare_dir FIG] [--external_dir EXT]
"""
import argparse
import hashlib
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from common import CLASSES, index_nickparvar, index_external, save_json

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _POP[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def dihedral(a):
    """8 rotations/reflections of a square array (last two axes)."""
    out = []
    for k in range(4):
        r = np.rot90(a, k, axes=(-2, -1))
        out.append(r)
        out.append(np.flip(r, axis=-1))
    return out


def phash_bits(g32):
    d = dctn(g32.astype(np.float64), norm="ortho")[:8, :8]
    med = np.median(d.flatten()[1:])
    bits = (d > med).flatten()
    return np.packbits(bits).view(">u8")[0].astype(np.uint64)


def load_gray(path, size):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)


def image_signatures(paths, label="images", array_loader=None):
    """Returns md5 list, base pHash (N,), dihedral pHashes (N,8), normalised thumbs (N,64,64)."""
    n = len(paths)
    md5s, h0, hv = [], np.zeros(n, np.uint64), np.zeros((n, 8), np.uint64)
    th = np.zeros((n, 64, 64), np.float32)
    t0 = time.time()
    for i, p in enumerate(paths):
        if array_loader is None:
            with Image.open(p) as im:
                g = im.convert("L")
                md5s.append(hashlib.md5(np.asarray(g).tobytes()).hexdigest())
                g32 = np.asarray(g.resize((32, 32), Image.LANCZOS), np.float32)
                g64 = np.asarray(g.resize((64, 64), Image.LANCZOS), np.float32)
        else:
            g = array_loader(p)  # 2-D uint8 array
            md5s.append(hashlib.md5(g.tobytes()).hexdigest())
            im = Image.fromarray(g)
            g32 = np.asarray(im.resize((32, 32), Image.LANCZOS), np.float32)
            g64 = np.asarray(im.resize((64, 64), Image.LANCZOS), np.float32)
        vs = dihedral(g32)
        h0[i] = phash_bits(vs[0])
        hv[i] = [phash_bits(v) for v in vs]
        z = g64 - g64.mean()
        th[i] = z / (np.linalg.norm(z) + 1e-8)
        if (i + 1) % 1000 == 0:
            print(f"  {label}: signatures {i+1}/{n} ({time.time()-t0:.0f}s)")
    return md5s, h0, hv, th


def candidate_pairs(h0_a, hv_b, thr, same_set, chunk=64):
    """Pairs (i in A, j in B) with min_k Hamming(h0_a[i], hv_b[j,k]) <= thr."""
    I, J = [], []
    for s in range(0, len(h0_a), chunk):
        x = h0_a[s:s + chunk, None, None] ^ hv_b[None, :, :]
        d = popcount64(x).min(-1)
        ii, jj = np.nonzero(d <= thr)
        ii = ii + s
        if same_set:
            keep = jj > ii
            ii, jj = ii[keep], jj[keep]
        I.append(ii)
        J.append(jj)
    return (np.concatenate(I) if I else np.array([], int)), (np.concatenate(J) if J else np.array([], int))


def ncc_max(th_a, th_b, I, J, batch=4096):
    out = np.zeros(len(I), np.float32)
    for s in range(0, len(I), batch):
        A = th_a[I[s:s + batch]]
        B = th_b[J[s:s + batch]]
        best = np.full(len(A), -1.0, np.float32)
        for v in dihedral(B):
            best = np.maximum(best, np.einsum("nij,nij->n", A, v))
        out[s:s + batch] = best
    return out


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def load_figshare(fig_dir):
    """Read Cheng et al. figshare .mat files (MATLAB v7.3 / HDF5)."""
    import h5py
    recs = []
    files = []
    for dp, _, fs in os.walk(fig_dir):
        files += [os.path.join(dp, f) for f in fs if f.lower().endswith(".mat")]
    for f in sorted(files):
        try:
            with h5py.File(f, "r") as h:
                cj = h["cjdata"]
                pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).flatten())
                img = np.array(cj["image"]).T.astype(np.float32)
            lo, hi = np.percentile(img, [0.5, 99.5])
            img8 = np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            recs.append({"fig_file": f, "pid": pid.strip(), "img": img8})
        except Exception as e:  # pragma: no cover
            print("  [warn] could not read", f, e)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--phash_thr", type=int, default=8)
    ap.add_argument("--ncc_thr", type=float, default=0.95)
    ap.add_argument("--figshare_dir", default=None,
                    help="folder with Cheng et al. figshare .mat files (optional, recovers patient IDs)")
    ap.add_argument("--external_dir", default=None,
                    help="external test dataset; checked for overlap with this dataset")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = index_nickparvar(a.data_root).reset_index(drop=True)
    n = len(df)
    print(f"computing signatures for {n} images")
    md5s, h0, hv, th = image_signatures(df["path"].tolist(), "dataset")
    df["md5"] = md5s

    uf = UF(n)
    links = []
    # exact duplicates
    for _, g in df.groupby("md5"):
        ix = g.index.tolist()
        for j in ix[1:]:
            uf.union(ix[0], j)
            links.append((ix[0], j, "exact", 1.0))
    n_exact_pairs = len(links)
    # near duplicates
    I, J = candidate_pairs(h0, hv, a.phash_thr, same_set=True)
    print(f"  pHash candidate pairs: {len(I)}")
    ncc = ncc_max(th, th, I, J)
    ok = ncc >= a.ncc_thr
    for i, j, c in zip(I[ok], J[ok], ncc[ok]):
        uf.union(int(i), int(j))
        links.append((int(i), int(j), "near", float(c)))
    print(f"  confirmed near-duplicate pairs (NCC>={a.ncc_thr}): {int(ok.sum())}")

    # patient recovery via figshare
    df["patient_id"] = ""
    pid_info = {"attempted": False}
    if a.figshare_dir:
        recs = load_figshare(a.figshare_dir)
        print(f"  figshare slices read: {len(recs)}")
        if recs:
            fig_arr = {r["fig_file"]: r["img"] for r in recs}
            _, fh0, fhv, fth = image_signatures([r["fig_file"] for r in recs], "figshare",
                                                array_loader=lambda p: fig_arr[p])
            FI, FJ = candidate_pairs(h0, fhv, a.phash_thr + 2, same_set=False)
            fn = ncc_max(th, fth, FI, FJ)
            best = {}
            for i, j, c in zip(FI, FJ, fn):
                if c >= a.ncc_thr and c > best.get(int(i), (None, -1))[1]:
                    best[int(i)] = (int(j), float(c))
            for i, (j, _) in best.items():
                df.at[i, "patient_id"] = recs[j]["pid"]
            # link all images of the same patient
            for pid, g in df[df.patient_id != ""].groupby("patient_id"):
                ix = g.index.tolist()
                for j in ix[1:]:
                    uf.union(ix[0], j)
                    links.append((ix[0], j, "patient", 1.0))
            pid_info = {"attempted": True, "figshare_slices": len(recs),
                        "images_with_patient_id": int((df.patient_id != "").sum()),
                        "n_patients": int(df.loc[df.patient_id != "", "patient_id"].nunique())}

    df["group_id"] = [uf.find(i) for i in range(n)]
    gsize = df.groupby("group_id")["rel"].transform("count")
    has_train = df.groupby("group_id")["split"].transform(lambda s: (s == "train").any())
    n_labels = df.groupby("group_id")["label"].transform("nunique")
    df["group_size"] = gsize
    df["test_has_train_twin"] = (df.split == "test") & has_train
    df["label_conflict_group"] = n_labels > 1

    te = df[df.split == "test"]
    tr = df[df.split == "train"]
    tr_in_dup = tr.groupby("group_id")["rel"].transform("count") > 1
    clean = te[~te.test_has_train_twin]
    summary = {
        "thresholds": {"phash_hamming": a.phash_thr, "ncc": a.ncc_thr},
        "exact_duplicate_pairs": n_exact_pairs,
        "near_duplicate_pairs": int(ok.sum()),
        "multi_image_groups": int((df.groupby("group_id").size() > 1).sum()),
        "images_in_multi_image_groups": int((gsize > 1).sum()),
        "pct_images_in_multi_image_groups": float((gsize > 1).mean() * 100),
        "train_images_with_within_train_twin": int(tr_in_dup.sum()),
        "pct_train_with_within_train_twin": float(tr_in_dup.mean() * 100),
        "test_images_with_train_twin": int(te.test_has_train_twin.sum()),
        "pct_test_with_train_twin": float(te.test_has_train_twin.mean() * 100),
        "test_with_train_twin_per_class": te.groupby("cls")["test_has_train_twin"].sum().reindex(CLASSES).astype(int).to_dict(),
        "label_conflict_groups": int(df.loc[df.label_conflict_group, "group_id"].nunique()),
        "label_conflict_images": int(df.label_conflict_group.sum()),
        "n_train_groups": int(tr.group_id.nunique()),
        "n_test_clean": int(len(clean)),
        "test_clean_per_class": clean.groupby("cls").size().reindex(CLASSES).fillna(0).astype(int).to_dict(),
        "patient_recovery": pid_info,
    }

    # external dataset overlap check
    if a.external_dir:
        ex = index_external(a.external_dir).reset_index(drop=True)
        print(f"checking external dataset overlap ({len(ex)} images)")
        _, eh0, ehv, eth = image_signatures(ex["path"].tolist(), "external")
        EI, EJ = candidate_pairs(eh0, hv, a.phash_thr, same_set=False)
        en = ncc_max(eth, th, EI, EJ)
        twin = np.zeros(len(ex), bool)
        twin[EI[en >= a.ncc_thr]] = True
        ex["has_twin_in_main_dataset"] = twin
        ex.drop(columns=["path"]).to_csv(os.path.join(a.out, "external_overlap.csv"), index=False)
        summary["external"] = {"n": int(len(ex)), "with_twin_in_main_dataset": int(twin.sum()),
                               "pct_with_twin": float(twin.mean() * 100),
                               "per_class": ex.groupby("cls").size().reindex(CLASSES).fillna(0).astype(int).to_dict()}

    df.drop(columns=["path"]).to_csv(os.path.join(a.out, "groups.csv"), index=False)
    pd.DataFrame([(df.rel[i], df.rel[j], df.split[i], df.split[j], df.cls[i], df.cls[j], k, c)
                  for i, j, k, c in links],
                 columns=["rel_a", "rel_b", "split_a", "split_b", "cls_a", "cls_b", "kind", "ncc"]
                 ).to_csv(os.path.join(a.out, "duplicate_links.csv"), index=False)
    save_json(summary, os.path.join(a.out, "dedup_summary.json"))
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
