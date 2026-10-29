"""Step 4 - shortcut probes, Grad-CAM and glioma error analysis

  a) occlusion probes   : accuracy when the image centre (brain) or only the
                          border is masked. High accuracy with the centre
                          masked = the model can use non-anatomical cues.
  b) metadata baseline  : classifier that sees ONLY file/acquisition properties
                          (size, format, colour mode, file size, border
                          intensity) - measures how separable classes are
                          without anatomy.
  c) Grad-CAM           : saliency for correct and misclassified test images.
  d) glioma errors      : properties of misclassified gliomas vs correctly
                          classified gliomas, plus an image gallery.

usage: python scripts/s04_explain.py --data_root DATA --groups results/groups.csv --out results \
          --model efficientnet_b0 --mode two_stage --seed 0
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from common import (CLASSES, CLASS_PRETTY, Config, build_cache, build_model, compute_metrics,
                    index_nickparvar, predict, save_json)


def occlude(imgs, kind, frac):
    x = np.array(imgs, copy=True)
    n, h, w, _ = x.shape
    ch, cw = int(h * frac), int(w * frac)
    t, l = (h - ch) // 2, (w - cw) // 2
    if kind == "centre":
        x[:, t:t + ch, l:l + cw, :] = 0
    else:  # keep only the centre
        m = np.zeros((h, w), bool)
        m[t:t + ch, l:l + cw] = True
        x[:, ~m, :] = 0
    return x


def file_features(paths):
    rows = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            g = np.asarray(im.convert("L"), np.float32)
            mode, fmt = im.mode, (im.format or "").upper()
        b = max(2, int(min(w, h) * 0.05))
        border = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
        rows.append({"width": w, "height": h, "aspect": w / h, "kbytes": os.path.getsize(p) / 1024,
                     "is_png": float(fmt == "PNG"), "is_jpeg": float(fmt in ("JPEG", "JPG")),
                     "mode_L": float(mode == "L"), "mode_RGB": float(mode == "RGB"),
                     "border_mean": float(border.mean()), "border_std": float(border.std()),
                     "frac_black": float((g < 5).mean())})
    return pd.DataFrame(rows)


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach()))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach()))

    def __call__(self, x, cls):
        import torch
        import torch.nn.functional as F
        self.model.zero_grad()
        out = self.model(x)
        out[0, cls].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()


def target_layer(model, name):
    if name == "efficientnet_b0":
        return model.features[-1]
    if name == "resnet18":
        return model.layer4[-1]
    if name == "mobilenet_v3_large":
        return model.features[-1]
    if name == "shufflenet_v2_x1_0":
        return model.conv5
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--mode", default="two_stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--gallery_n", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    cfg = Config()
    if a.smoke:
        cfg.smoke()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(a.out, "explain")
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = index_nickparvar(a.data_root)
    g = pd.read_csv(a.groups)[["rel", "group_id", "test_has_train_twin"]]
    df = df.merge(g, on="rel", how="left")
    tr = df[df.split == "train"].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    clean = ~te["test_has_train_twin"].astype(bool).values
    te_imgs = build_cache(te["path"].tolist(), cfg.img_size, os.path.join(a.cache_dir, "test"))
    y_te = te["label"].values

    tag = f"{a.model}__{a.mode}__s{a.seed}"
    model, _ = build_model(a.model, cfg)
    model.load_state_dict(torch.load(os.path.join(a.out, "test", tag + ".pt"), map_location="cpu"))
    model.to(dev).eval()

    res = {}
    # a) occlusion probes on the clean test set
    ci = np.where(clean)[0]
    base = compute_metrics(y_te[ci], predict(model, te_imgs[ci], cfg, dev))
    occ_c = compute_metrics(y_te[ci], predict(model, occlude(te_imgs[ci], "centre", 0.6), cfg, dev))
    occ_b = compute_metrics(y_te[ci], predict(model, occlude(te_imgs[ci], "border", 0.6), cfg, dev))
    res["occlusion"] = {"baseline_acc": base["accuracy"], "centre_masked_acc": occ_c["accuracy"],
                        "centre_masked_bacc": occ_c["balanced_accuracy"],
                        "border_masked_acc": occ_b["accuracy"], "chance": 1 / len(CLASSES),
                        "mask_fraction": 0.6}
    print("occlusion:", res["occlusion"])

    # b) metadata-only classifier (train partition -> clean test)
    ftr, fte = file_features(tr["path"]), file_features(te["path"])
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    rf.fit(ftr.values, tr["label"].values)
    pm = rf.predict(fte.values[ci])
    tumour_tr = (tr["cls"] != "notumor").astype(int).values
    tumour_te = (te["cls"] != "notumor").astype(int).values[ci]
    rf2 = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(ftr.values, tumour_tr)
    res["metadata_only"] = {
        "four_class_balanced_acc": float(balanced_accuracy_score(y_te[ci], pm)),
        "tumour_vs_no_tumour_balanced_acc": float(balanced_accuracy_score(tumour_te, rf2.predict(fte.values[ci]))),
        "features": list(ftr.columns),
        "importances": dict(zip(ftr.columns, rf.feature_importances_.round(4).tolist()))}
    print("metadata-only:", res["metadata_only"])

    # property comparison train vs test (per class)
    ftr["split"], fte["split"] = "train", "test"
    ftr["cls"], fte["cls"] = tr["cls"].values, te["cls"].values
    props = pd.concat([ftr, fte])
    props.groupby(["cls", "split"]).median(numeric_only=True).round(3).to_csv(
        os.path.join(out_dir, "image_properties_train_vs_test.csv"))

    # d) glioma error analysis on the official test predictions
    pred_csv = os.path.join(a.out, "test", f"{tag}__official_predictions.csv")
    pr = pd.read_csv(pred_csv)
    fte_idx = fte.reset_index(drop=True)
    pr = pd.concat([pr.reset_index(drop=True), fte_idx.drop(columns=["split", "cls"])], axis=1)
    pr["clean"] = clean
    gl = pr[pr.true == "glioma"]
    gl_summary = gl.groupby("pred").agg(n=("rel", "count"), median_width=("width", "median"),
                                        median_kbytes=("kbytes", "median"),
                                        median_border=("border_mean", "median"),
                                        pct_png=("is_png", "mean"), pct_clean=("clean", "mean"),
                                        median_conf=("confidence", "median"))
    gl_summary.to_csv(os.path.join(out_dir, "glioma_errors_by_prediction.csv"))
    res["glioma_errors"] = gl_summary.reset_index().to_dict(orient="records")
    print(gl_summary)

    # c) Grad-CAM figures
    cam = GradCAM(model, target_layer(model, a.model))
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]

    def to_tensor(img):
        x = (img.transpose(2, 0, 1) / 255.0 - mean) / std
        return torch.tensor(x, dtype=torch.float32)[None].to(dev)

    def panel(indices, title, fname, ncols=4):
        if len(indices) == 0:
            return
        n = len(indices)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 3.2, nrows * 1.7))
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            ax.axis("off")
        for k, i in enumerate(indices):
            r, c = divmod(k, ncols)
            img = te_imgs[i]
            pc = CLASSES.index(pr.loc[i, "pred"])
            heat = cam(to_tensor(img), pc)
            axes[r, 2 * c].imshow(img)
            axes[r, 2 * c].set_title(f"true {CLASS_PRETTY[pr.loc[i,'true']]}\npred {CLASS_PRETTY[pr.loc[i,'pred']]}", fontsize=6)
            axes[r, 2 * c + 1].imshow(img)
            axes[r, 2 * c + 1].imshow(heat, cmap="jet", alpha=0.45)
            axes[r, 2 * c + 1].set_title("Grad-CAM", fontsize=6)
        fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, fname), dpi=200)
        plt.close(fig)

    rng = np.random.default_rng(0)
    correct = []
    for c in CLASSES:
        ix = pr.index[(pr.true == c) & (pr.pred == c) & pr.clean].tolist()
        correct += list(rng.choice(ix, min(2, len(ix)), replace=False)) if ix else []
    panel(correct, "Correctly classified clean-test images (2 per class)", "fig5_gradcam_correct.png")
    g2n = pr.index[(pr.true == "glioma") & (pr.pred == "notumor")].tolist()[:a.gallery_n]
    g2m = pr.index[(pr.true == "glioma") & (pr.pred == "meningioma")].tolist()[:a.gallery_n]
    panel(g2n, "Gliomas predicted as no tumour", "fig6a_glioma_to_notumour.png")
    panel(g2m, "Gliomas predicted as meningioma", "fig6b_glioma_to_meningioma.png")
    save_json(res, os.path.join(out_dir, "explain_summary.json"))
    print("wrote", os.path.join(out_dir, "explain_summary.json"))


if __name__ == "__main__":
    main()
