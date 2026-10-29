"""Step 7 - analysis by duplicate status and image source.

Uses only files written by steps 0-6 (no images needed):
  * official-test performance of the final EfficientNet-B0 models split by
    twin status (NCC >= 0.95 or same patient; NCC >= 0.98) and by glioma
    source (figshare-matched vs other)            -> table 8, figure 5
  * metadata-only random forest (width, height, aspect ratio, file size,
    file format, colour mode): grouped 5-fold CV on the training partition
    and train->official test                       -> section 4.6
  * figure 4 (confusion matrices) and figure 7 (reliability diagram)

usage: python scripts/s07_source_analysis.py --out results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold

    R = a.out
    fig_dir = os.path.join(R, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    g95 = pd.read_csv(os.path.join(R, "groups.csv")).set_index("rel")
    g98 = pd.read_csv(os.path.join(R, "dedup_098", "groups.csv")).set_index("rel")
    figshare = g95.patient_id.fillna("") != ""

    # ---------------- source composition (table 2)
    comp = pd.crosstab([g95.cls, g95.split], figshare.rename("figshare_source"))
    comp.to_csv(os.path.join(R, "source_composition.csv"))

    # ---------------- subsets of the official test partition (table 8)
    subs = {}
    for s in a.seeds:
        p = pd.read_csv(os.path.join(R, "test", f"efficientnet_b0__two_stage__s{s}__official_predictions.csv")).set_index("rel")
        p["tw95"] = g95.loc[p.index, "test_has_train_twin"].astype(bool).values
        p["tw98"] = g98.loc[p.index, "test_has_train_twin"].astype(bool).values
        p["fig"] = figshare.loc[p.index].values
        S = {"all": p, "with_twin_095": p[p.tw95], "without_twin_095": p[~p.tw95],
             "without_neardup_098": p[~p.tw98],
             "gliomas_figshare": p[(p.true == "glioma") & p.fig],
             "gliomas_other_source": p[(p.true == "glioma") & ~p.fig]}
        for k, d in S.items():
            gl = d[d.true == "glioma"]
            subs.setdefault(k, []).append({
                "n": len(d), "acc": (d.true == d.pred).mean(), "gl_n": len(gl),
                "gl_recall": (gl.pred == "glioma").mean() if len(gl) else np.nan,
                "gl_to_notumor": (gl.pred == "notumor").mean() if len(gl) else np.nan,
                "gl_to_meningioma": (gl.pred == "meningioma").mean() if len(gl) else np.nan})
        if s == a.seeds[0]:
            gl = p[p.true == "glioma"]
            err = gl[gl.pred != "glioma"]
            g2n = err[err.pred == "notumor"]
            extra = {"seed": s, "glioma_errors": int(len(err)), "glioma_errors_other_source": int((~err.fig).sum()),
                     "glioma_to_notumor": int(len(g2n)), "glioma_to_notumor_other_source": int((~g2n.fig).sum()),
                     "glioma_to_notumor_median_confidence": float(g2n.confidence.median()) if len(g2n) else None}
    table = {}
    for k, v in subs.items():
        df = pd.DataFrame(v)
        table[k] = {"n": int(df.n[0]), "gl_n": int(df.gl_n[0])}
        for c in ["acc", "gl_recall", "gl_to_notumor", "gl_to_meningioma"]:
            table[k][c] = None if df[c].isna().any() else [float(100 * df[c].mean()), float(100 * df[c].std(ddof=1))]
    # accuracy without the other-source gliomas
    accs = []
    for s in a.seeds:
        p = pd.read_csv(os.path.join(R, "test", f"efficientnet_b0__two_stage__s{s}__official_predictions.csv")).set_index("rel")
        keep = ~((p.true == "glioma") & ~figshare.loc[p.index].values)
        accs.append((p.true == p.pred)[keep].mean())
    table["acc_excluding_other_source_gliomas"] = [float(100 * np.mean(accs)), float(100 * np.std(accs, ddof=1))]
    table["seed0_details"] = extra

    # ---------------- metadata-only classifier
    au = pd.read_csv(os.path.join(R, "audit_index.csv"))
    d = g95.reset_index().merge(au[["rel", "width", "height", "mode", "format", "bytes"]], on="rel")
    X = pd.DataFrame({"width": d.width, "height": d.height, "aspect": d.width / d.height, "kbytes": d.bytes / 1024,
                      "is_png": (d.format == "PNG").astype(float), "is_jpeg": (d.format == "JPEG").astype(float),
                      "mode_L": (d["mode"] == "L").astype(float), "mode_RGB": (d["mode"] == "RGB").astype(float)})
    y, yb = d.label.values, (d.cls != "notumor").astype(int).values
    tr = (d.split == "train").values
    b4, b2 = [], []
    for tri, tei in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X[tr], y[tr], d.group_id[tr]):
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr].iloc[tri], y[tr][tri])
        b4.append(balanced_accuracy_score(y[tr][tei], rf.predict(X[tr].iloc[tei])))
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr].iloc[tri], yb[tr][tri])
        b2.append(balanced_accuracy_score(yb[tr][tei], rf.predict(X[tr].iloc[tei])))
    rf4 = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
    rf2 = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X[tr], yb[tr])
    p4 = rf4.predict(X[~tr])
    te = d[~tr].copy()
    te["p4"] = p4
    meta = {"grouped_cv_4class_bacc": [100 * np.mean(b4), 100 * np.std(b4, ddof=1)],
            "grouped_cv_binary_bacc": [100 * np.mean(b2), 100 * np.std(b2, ddof=1)],
            "test_4class_bacc": 100 * balanced_accuracy_score(y[~tr], p4),
            "test_binary_bacc": 100 * balanced_accuracy_score(yb[~tr], rf2.predict(X[~tr])),
            "test_other_source_glioma_recall": 100 * float((te[(te.cls == "glioma") & (te.patient_id.fillna("") == "")].p4 == 0).mean()),
            "importances": dict(zip(X.columns, rf4.feature_importances_.round(3).tolist())),
            "pct_512_train_by_class": (d[tr].assign(is512=(d.width == 512) & (d.height == 512))
                                       .groupby("cls").is512.mean() * 100).round(1).to_dict()}
    table["metadata_only"] = meta
    with open(os.path.join(R, "source_analysis.json"), "w") as fh:
        json.dump(table, fh, indent=1, default=float)

    # ---------------- figures
    plt.rcParams.update({"font.size": 8})
    keys = ["with_twin_095", "without_twin_095", "gliomas_figshare", "gliomas_other_source"]
    met = ["acc", "acc", "gl_recall", "gl_recall"]
    if not figshare.any():
        print("WARNING: no figshare patient IDs were recovered (run s01_dedup.py with --figshare_dir); "
              "the source-specific bars will be empty.")
    vals = [table[k][m][0] if table[k][m] else 0.0 for k, m in zip(keys, met)]
    errs = [table[k][m][1] if table[k][m] else 0.0 for k, m in zip(keys, met)]
    ns = [table[keys[0]]["n"], table[keys[1]]["n"], table[keys[2]]["gl_n"], table[keys[3]]["gl_n"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(range(4), vals, yerr=errs, capsize=4, color=["#1F4E79", "#C0392B", "#1F4E79", "#C0392B"], width=0.62)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + errs[i] + 2.5, f"{v:.1f}%\n(n = {n:,})" if n else "n/a", ha="center", fontsize=7.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["With\ntraining twin", "Without\ntraining twin", "Gliomas,\nfigshare source", "Gliomas,\nother sources"])
    ax.set_ylim(0, 135); ax.set_yticks(range(0, 101, 20)); ax.set_ylabel("Accuracy / glioma recall (%)")
    ax.axvline(1.5, color="grey", lw=0.6, ls=":")
    ax.text(0.5, 126, "Accuracy", ha="center", fontweight="bold"); ax.text(2.5, 126, "Glioma recall", ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_source_effect.png"), dpi=250); plt.close(fig)

    CL = ["Glioma", "Meningioma", "No tumour", "Pituitary"]
    r = json.load(open(os.path.join(R, "test", f"efficientnet_b0__two_stage__s{a.seeds[0]}.json")))
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, (k, t) in zip(axes, [("official", "Official test partition (n = 1,600)"),
                                 ("clean", f"Test images without a training twin (n = {table['without_twin_095']['n']})")]):
        cm = np.array(r["sets"][k]["confusion"]); rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CL, rotation=30, ha="right", fontsize=6.5); ax.set_yticklabels(CL, fontsize=6.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{100 * rn[i, j]:.1f}%" if cm[i].sum() else "—", ha="center", va="center",
                        fontsize=5.8, color="white" if rn[i, j] > 0.6 else "black")
        ax.set_title(t, fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_test_confusion.png"), dpi=250); plt.close(fig)

    p = pd.read_csv(os.path.join(R, "test", f"efficientnet_b0__two_stage__s{a.seeds[0]}__clean_predictions.csv"))
    conf, corr = p.confidence.values, (p.true == p.pred).values
    xs, ys, nn = [], [], []
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(corr[m].mean()); nn.append(int(m.sum()))
    ece = sum(((conf > lo) & (conf <= hi)).mean() * abs(corr[(conf > lo) & (conf <= hi)].mean() - conf[(conf > lo) & (conf <= hi)].mean())
              for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]) if ((conf > lo) & (conf <= hi)).any())
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1); ax.plot(xs, ys, "o-", color="#C0392B", ms=4)
    for x, yv, n in zip(xs, ys, nn):
        ax.annotate(str(n), (x, yv), textcoords="offset points", xytext=(4, -9), fontsize=6)
    ax.set_xlabel("Mean predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Test images without a training twin\n(seed {a.seeds[0]}, n = {len(p)}, ECE = {ece:.3f})", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_reliability.png"), dpi=250); plt.close(fig)
    print(json.dumps({k: v for k, v in table.items() if k != "metadata_only"}, indent=1, default=float)[:2500])
    print("metadata-only:", {k: v for k, v in meta.items() if k != "importances"})


if __name__ == "__main__":
    main()
