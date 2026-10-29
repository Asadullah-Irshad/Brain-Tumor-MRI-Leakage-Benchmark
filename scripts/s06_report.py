"""Step 6 - aggregate all results into the numbers, tables and figures used in
the paper.

Writes:
  results/manuscript_values.json   every summary number reported in the paper (key -> value)
  results/manuscript_values.md     the same, human-readable
  results/figures/*.png            figures 1-4 and the duplicate examples

Statistics (as reported in the paper):
  * cross-validation: mean +/- SD over all (seed x fold) runs; 95% CI of the
    mean from the Nadeau-Bengio corrected resampled t (accounts for
    overlapping training sets). Nothing is resampled at image level here.
  * test sets: mean +/- SD across the independently trained final models
    (seeds), plus a 95% percentile bootstrap CI (2,000 class-stratified
    resamples of TEST IMAGES) for the seed-0 model.

usage: python scripts/s06_report.py --out results [--data_root DATA]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from common import CLASSES, CLASS_PRETTY, MODEL_PRETTY, corrected_t_ci, load_json, save_json

CODE = {"GLI": "glioma", "MEN": "meningioma", "NOT": "notumor", "PIT": "pituitary"}
CONFIGS = {"EFF2": ("efficientnet_b0", "two_stage"), "EFFF": ("efficientnet_b0", "frozen"),
           "EFF1": ("efficientnet_b0", "single_stage"), "R18": ("resnet18", "two_stage"),
           "MV3": ("mobilenet_v3_large", "two_stage"), "SHF": ("shufflenet_v2_x1_0", "two_stage")}


def pct(x, d=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def f3(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def pm_pct(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "n/a"
    return f"{100*v.mean():.1f} ± {100*v.std(ddof=1):.1f}" if len(v) > 1 else f"{100*v.mean():.1f}"


def pm3(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def rng_txt(lo, hi, f=pct):
    return f"{f(lo)}–{f(hi)}"


def comma(n):
    return f"{int(n):,}"


def load_runs(pattern):
    runs = []
    for f in sorted(glob.glob(pattern)):
        if f.endswith("run_info.json"):
            continue
        runs.append(load_json(f))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--data_root", default=None, help="needed only for the duplicate-example figure")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig_dir = os.path.join(a.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    V = {}

    # ---------------- audit
    au = os.path.join(a.out, "audit_summary.json")
    if os.path.exists(au):
        s = load_json(au)
        V["RELEASE"] = s["release_detected"]
        V["N_TRAIN"], V["N_TEST"], V["N_TOTAL"] = comma(s["n_train"]), comma(s["n_test"]), comma(s["n_total"])
        for sp, key in [("train_per_class", "TRAIN"), ("test_per_class", "TEST")]:
            for c3, c in CODE.items():
                V[f"{key}_{c3}"] = comma(s[sp][c])

    # ---------------- dedup
    du = os.path.join(a.out, "dedup_summary.json")
    if os.path.exists(du):
        d = load_json(du)
        V.update({"DUP_EXACT_PAIRS": comma(d["exact_duplicate_pairs"]),
                  "DUP_NEAR_PAIRS": comma(d["near_duplicate_pairs"]),
                  "DUP_GROUPS": comma(d["multi_image_groups"]),
                  "DUP_IMG_N": comma(d["images_in_multi_image_groups"]),
                  "DUP_IMG_PCT": f"{d['pct_images_in_multi_image_groups']:.1f}",
                  "TRAIN_TWIN_N": comma(d["train_images_with_within_train_twin"]),
                  "TRAIN_TWIN_PCT": f"{d['pct_train_with_within_train_twin']:.1f}",
                  "TEST_TWIN_N": comma(d["test_images_with_train_twin"]),
                  "TEST_TWIN_PCT": f"{d['pct_test_with_train_twin']:.1f}",
                  "LABEL_CONFLICT_GROUPS": comma(d["label_conflict_groups"]),
                  "LABEL_CONFLICT_IMAGES": comma(d["label_conflict_images"]),
                  "N_TRAIN_GROUPS": comma(d["n_train_groups"]),
                  "N_TEST_CLEAN": comma(d["n_test_clean"]),
                  "PHASH_THR": str(d["thresholds"]["phash_hamming"]),
                  "NCC_THR": f"{d['thresholds']['ncc']:.2f}"})
        for c3, c in CODE.items():
            V[f"TWIN_{c3}"] = comma(d["test_with_train_twin_per_class"][c])
            V[f"CLEAN_{c3}"] = comma(d["test_clean_per_class"][c])
        pr = d["patient_recovery"]
        if pr.get("attempted"):
            V["PID_IMAGES"] = comma(pr["images_with_patient_id"])
            V["PID_PATIENTS"] = comma(pr["n_patients"])
        else:
            V["PID_IMAGES"], V["PID_PATIENTS"] = "0", "0"
        d98 = os.path.join(a.out, "dedup_098", "dedup_summary.json")
        if os.path.exists(d98):  # sensitivity analysis with the stricter NCC threshold
            s98 = load_json(d98)
            V["TEST_TWIN_N_098"] = comma(s98["test_images_with_train_twin"])
            V["TEST_TWIN_PCT_098"] = f"{s98['pct_test_with_train_twin']:.1f}"
        if "external" in d:
            e = d["external"]
            V["EXT_N_RAW"] = comma(e["n"])
            V["EXT_TWIN_N"] = comma(e["with_twin_in_main_dataset"])
            V["EXT_TWIN_PCT"] = f"{e['pct_with_twin']:.1f}"

    # ---------------- cross-validation
    cv_all = {}
    for proto in ["grouped", "naive"]:
        runs = load_runs(os.path.join(a.out, "cv", proto, "*__s*_f*.json"))
        for r in runs:
            cv_all.setdefault((proto, r["model"], r["mode"]), []).append(r)

    def cv_block(prefix, runs, ratio=0.25):
        acc = [r["accuracy"] for r in runs]
        f1 = [r["macro_f1"] for r in runs]
        auc = [r["macro_auc"] for r in runs]
        m, sd, ci = corrected_t_ci(acc, ratio)
        V[f"{prefix}_NRUNS"] = str(len(runs))
        V[f"{prefix}_NSEEDS"] = str(len({r["seed"] for r in runs}))
        V[f"{prefix}_ACC"] = pm_pct(acc)
        V[f"{prefix}_ACC_CI"] = rng_txt(*ci)
        V[f"{prefix}_ACC_RANGE"] = rng_txt(min(acc), max(acc))
        V[f"{prefix}_BACC"] = pm_pct([r["balanced_accuracy"] for r in runs])
        V[f"{prefix}_F1"] = pm3(f1)
        V[f"{prefix}_F1_CI"] = rng_txt(*corrected_t_ci(f1, ratio)[2], f=f3)
        V[f"{prefix}_AUC"] = pm3(auc)
        V[f"{prefix}_AUC_CI"] = rng_txt(*corrected_t_ci(auc, ratio)[2], f=f3)
        V[f"{prefix}_ECE"] = pm3([r["ece"] for r in runs])
        for c3, c in CODE.items():
            for k, kk in [("precision", "P"), ("recall", "R"), ("f1", "F1")]:
                V[f"{prefix}_{c3}_{kk}"] = pm3([r["per_class"][c][k] for r in runs])
        return acc

    main_runs = cv_all.get(("grouped", "efficientnet_b0", "two_stage"), [])
    if main_runs:
        cv_block("CV", main_runs)
    naive_runs = cv_all.get(("naive", "efficientnet_b0", "two_stage"), [])
    if naive_runs:
        cv_block("CVN", naive_runs)
        if main_runs:
            # compare on the seeds both protocols share
            seeds = {r["seed"] for r in naive_runs}
            g = np.mean([r["accuracy"] for r in main_runs if r["seed"] in seeds])
            n = np.mean([r["accuracy"] for r in naive_runs])
            V["LEAK_GAP_ACC"] = f"{100*(n-g):+.1f}"
    for code, (mdl, mode) in CONFIGS.items():
        runs = cv_all.get(("grouped", mdl, mode), [])
        if runs:
            V[f"B_{code}_ACC"] = pm_pct([r["accuracy"] for r in runs])
            V[f"B_{code}_F1"] = pm3([r["macro_f1"] for r in runs])
            V[f"B_{code}_AUC"] = pm3([r["macro_auc"] for r in runs])
            V[f"B_{code}_GLI_R"] = pm3([r["per_class"]["glioma"]["recall"] for r in runs])
            V[f"B_{code}_N"] = str(len(runs))
            V[f"B_{code}_PARAMS"] = f"{runs[0]['params']/1e6:.2f}"

    # Figure 1: per-run accuracy for every configuration
    rows = []
    for (proto, mdl, mode), runs in cv_all.items():
        lab = f"{MODEL_PRETTY[mdl]}\n{mode.replace('_', '-')}" + ("\n(image-level split)" if proto == "naive" else "")
        for r in runs:
            rows.append({"config": lab, "acc": 100 * r["accuracy"], "seed": r["seed"], "proto": proto})
    if rows:
        cdf = pd.DataFrame(rows)
        order = cdf.groupby("config")["acc"].mean().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(max(4.5, 1.15 * len(order)), 3.2))
        for i, c in enumerate(order):
            vals = cdf[cdf.config == c]
            jitter = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.25
            colors = ["#C0392B" if p == "naive" else "#1F4E79" for p in vals.proto]
            ax.scatter(i + jitter, vals.acc, s=12, c=colors, alpha=0.75, zorder=3)
            m, sd = vals.acc.mean(), vals.acc.std(ddof=1) if len(vals) > 1 else 0
            ax.errorbar(i + 0.3, m, yerr=sd, fmt="D", color="black", ms=4, capsize=3, zorder=4)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=6.5)
        ax.set_ylabel("Accuracy on held-out fold (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Cross-validation: every seed × fold run (dots); mean ± SD (diamond)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig1_cv_runs.png"), dpi=220)
        plt.close(fig)

    # Figure 2: pooled CV confusion matrix (main config, seed 0)
    s0 = [r for r in main_runs if r["seed"] == min(r2["seed"] for r2 in main_runs)] if main_runs else []
    if s0:
        cm = np.sum([np.array(r["confusion"]) for r in s0], 0)
        _confusion_fig(plt, cm, "Grouped cross-validation (pooled folds, seed 0)",
                       os.path.join(fig_dir, "fig2_cv_confusion.png"))

    # ---------------- test sets
    tests = load_runs(os.path.join(a.out, "test", "*__s*.json"))
    by_cfg = {}
    for r in tests:
        by_cfg.setdefault((r["model"], r["mode"]), []).append(r)
    main_t = sorted(by_cfg.get(("efficientnet_b0", "two_stage"), []), key=lambda r: r["seed"])
    tinfo = os.path.join(a.out, "test", "run_info.json")
    if os.path.exists(tinfo):
        ti = load_json(tinfo)
        V["EXT_NAME"] = ti.get("external_name") or "n/a"
        env = ti["env"]
        V.update({"SW_PY": env.get("python", "n/a"), "SW_TORCH": env.get("torch", "n/a"),
                  "SW_TV": env.get("torchvision", "n/a"), "HW_GPU": env.get("gpu", "n/a")})
    for sname, P in [("official", "T"), ("clean", "TC"), ("external", "TE")]:
        rs = [r for r in main_t if sname in r["sets"]]
        if not rs:
            continue
        ms = [r["sets"][sname] for r in rs]
        m0 = ms[0]
        V[f"{P}_N"] = comma(m0["n"])
        V[f"{P}_NSEEDS"] = str(len(ms))
        V[f"{P}_ACC"] = pm_pct([m["accuracy"] for m in ms])
        V[f"{P}_BACC"] = pm_pct([m["balanced_accuracy"] for m in ms])
        V[f"{P}_F1"] = pm3([m["macro_f1"] for m in ms])
        V[f"{P}_AUC"] = pm3([m["macro_auc"] for m in ms])
        V[f"{P}_ECE"] = pm3([m["ece"] for m in ms])
        b = m0["bootstrap_ci_95"]
        V[f"{P}_ACC_CI"] = rng_txt(*b["accuracy"])
        V[f"{P}_F1_CI"] = rng_txt(*b["macro_f1"], f=f3)
        V[f"{P}_AUC_CI"] = rng_txt(*b["macro_auc"], f=f3)
        for c3, c in CODE.items():
            for k, kk in [("precision", "P"), ("recall", "R"), ("f1", "F1")]:
                V[f"{P}_{c3}_{kk}"] = pm3([m["per_class"][c][k] for m in ms])
        cm = np.array(m0["confusion"])
        gi, mi, ni = CLASSES.index("glioma"), CLASSES.index("meningioma"), CLASSES.index("notumor")
        ng = cm[gi].sum()
        tum = [i for i in range(4) if i != ni]
        V[f"{P}_GLI_N"] = comma(ng)
        V[f"{P}_GLI2NOT_N"] = comma(cm[gi, ni])
        V[f"{P}_GLI2NOT_PCT"] = pct(cm[gi, ni] / max(ng, 1))
        V[f"{P}_GLI2MEN_N"] = comma(cm[gi, mi])
        V[f"{P}_GLI2MEN_PCT"] = pct(cm[gi, mi] / max(ng, 1))
        V[f"{P}_TUM2NOT_N"] = comma(cm[tum, ni].sum())
        V[f"{P}_TUM2NOT_PCT"] = pct(cm[tum, ni].sum() / max(cm[tum].sum(), 1))
        V[f"{P}_NOT2TUM_N"] = comma(cm[ni, tum].sum())
        V[f"{P}_NOT2TUM_PCT"] = pct(cm[ni, tum].sum() / max(cm[ni].sum(), 1))
    if main_runs and main_t and "clean" in main_t[0]["sets"]:
        cvg = np.mean([r["per_class"]["glioma"]["recall"] for r in main_runs])
        tg = np.mean([r["sets"]["clean"]["per_class"]["glioma"]["recall"] for r in main_t])
        V["GLI_R_DROP"] = f"{100*(cvg-tg):.1f}"
    for code, (mdl, mode) in CONFIGS.items():
        rs = by_cfg.get((mdl, mode), [])
        rs = [r for r in rs if "clean" in r["sets"]]
        if rs:
            V[f"BT_{code}_ACC"] = pm_pct([r["sets"]["clean"]["accuracy"] for r in rs])
            V[f"BT_{code}_F1"] = pm3([r["sets"]["clean"]["macro_f1"] for r in rs])
            V[f"BT_{code}_GLI_R"] = pm3([r["sets"]["clean"]["per_class"]["glioma"]["recall"] for r in rs])
            if any("external" in r["sets"] for r in rs):
                V[f"BT_{code}_EXT_ACC"] = pm_pct([r["sets"]["external"]["accuracy"] for r in rs if "external" in r["sets"]])

    # Figure 3: test confusion matrices (seed 0)
    if main_t:
        avail = [s for s in ["official", "clean", "external"] if s in main_t[0]["sets"]]
        fig, axes = plt.subplots(1, len(avail), figsize=(3.1 * len(avail), 2.9))
        axes = np.atleast_1d(axes)
        titles = {"official": "Official test partition", "clean": "Clean test (no training twin)",
                  "external": "External test"}
        for ax, s in zip(axes, avail):
            _confusion_ax(ax, np.array(main_t[0]["sets"][s]["confusion"]), titles[s])
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "fig3_test_confusion.png"), dpi=220)
        plt.close(fig)
        # Figure 4: reliability diagram on clean test, seed 0
        pf = glob.glob(os.path.join(a.out, "test", f"efficientnet_b0__two_stage__s{main_t[0]['seed']}__clean_predictions.csv"))
        if pf:
            pdf = pd.read_csv(pf[0])
            conf = pdf.confidence.values
            corr = (pdf.true == pdf.pred).values
            bins = np.linspace(0, 1, 11)
            xs, ys, ns = [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mk = (conf > lo) & (conf <= hi)
                if mk.sum():
                    xs.append(conf[mk].mean()); ys.append(corr[mk].mean()); ns.append(mk.sum())
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
            ax.plot(xs, ys, "o-", color="#1F4E79", ms=4)
            ax.set_xlabel("Mean predicted confidence")
            ax.set_ylabel("Observed accuracy")
            ax.set_title(f"Reliability, clean test (ECE {main_t[0]['sets']['clean']['ece']:.3f})", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "fig4_reliability.png"), dpi=220)
            plt.close(fig)

    # ---------------- explain
    ex = os.path.join(a.out, "explain", "explain_summary.json")
    if os.path.exists(ex):
        e = load_json(ex)
        V["OCC_BASE_ACC"] = pct(e["occlusion"]["baseline_acc"])
        V["OCC_CENTRE_ACC"] = pct(e["occlusion"]["centre_masked_acc"])
        V["OCC_BORDER_ACC"] = pct(e["occlusion"]["border_masked_acc"])
        V["META_BACC4"] = pct(e["metadata_only"]["four_class_balanced_acc"])
        V["META_BACC2"] = pct(e["metadata_only"]["tumour_vs_no_tumour_balanced_acc"])

    # ---------------- latency
    la = os.path.join(a.out, "latency.json")
    if os.path.exists(la):
        L = load_json(la)
        V["LAT_CPU"] = L["env"].get("cpu_model", "n/a")
        V["LAT_THREADS"] = str(L["torch_threads"])
        V["LAT_TORCH"] = L["env"].get("torch", "n/a")
        e = L["models"].get("efficientnet_b0", {})
        if e:
            V["LAT_PT_MS"] = f"{e['pytorch_ms_median']:.1f}"
            V["LAT_PT_P95"] = f"{e['pytorch_ms_p95']:.1f}"
            V["LAT_B32_MS"] = f"{e['pytorch_batch32_ms_per_image']:.1f}"
            V["LAT_ORT_MS"] = f"{e['onnxruntime_ms_median']:.1f}" if e.get("onnxruntime_ms_median") else "n/a"
            V["LAT_ORT_VER"] = e.get("onnxruntime_version", "n/a")
            V["MODEL_PARAMS"] = f"{e['params']/1e6:.2f}"
            V["MODEL_MB"] = f"{e['weights_mb']:.1f}"
            V["MODEL_GFLOPS"] = f"{e['gflops']:.2f}" if e.get("gflops") else "n/a"
        for code, (mdl, _) in CONFIGS.items():
            m = L["models"].get(mdl)
            if m:
                V[f"B_{code}_LAT"] = f"{m['pytorch_ms_median']:.1f}"
                V[f"B_{code}_GFLOPS"] = f"{m['gflops']:.2f}" if m.get("gflops") else "n/a"
                V[f"B_{code}_MB"] = f"{m['weights_mb']:.1f}"

    # duplicate example figure
    dl = os.path.join(a.out, "duplicate_links.csv")
    if a.data_root and os.path.exists(dl):
        from PIL import Image
        L = pd.read_csv(dl)
        cross = L[(L.split_a != L.split_b) & (L.kind != "patient")].sort_values("ncc").head(200)
        if len(cross):
            pick = cross.sample(min(4, len(cross)), random_state=0)
            fig, axes = plt.subplots(2, len(pick), figsize=(2.0 * len(pick), 4.1))
            axes = np.atleast_2d(axes).reshape(2, -1)
            for k, (_, r) in enumerate(pick.iterrows()):
                for row, (rel, sp, c) in enumerate([(r.rel_a, r.split_a, r.cls_a), (r.rel_b, r.split_b, r.cls_b)]):
                    with Image.open(os.path.join(a.data_root, rel)) as im:
                        axes[row, k].imshow(im.convert("L"), cmap="gray")
                    axes[row, k].set_title(f"{sp}: {CLASS_PRETTY[c]}\n{os.path.basename(rel)}", fontsize=6)
                    axes[row, k].axis("off")
                axes[1, k].text(0.5, -0.08, f"NCC {r.ncc:.3f}", transform=axes[1, k].transAxes,
                                ha="center", va="top", fontsize=6)
            fig.suptitle("Examples of training/test near-duplicates", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "figS1_duplicate_examples.png"), dpi=200)
            plt.close(fig)

    save_json(V, os.path.join(a.out, "manuscript_values.json"))
    with open(os.path.join(a.out, "manuscript_values.md"), "w") as fh:
        fh.write("| placeholder | value |\n|---|---|\n")
        for k in sorted(V):
            fh.write(f"| {{{{{k}}}}} | {V[k]} |\n")
    print(f"wrote {len(V)} values to {os.path.join(a.out, 'manuscript_values.json')}")


def _confusion_ax(ax, cm, title):
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    labs = [CLASS_PRETTY[c] for c in CLASSES]
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=6.5)
    ax.set_yticklabels(labs, fontsize=6.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{100*rown[i, j]:.1f}%", ha="center", va="center", fontsize=5.8,
                    color="white" if rown[i, j] > 0.6 else "black")
    ax.set_title(title, fontsize=7.5)


def _confusion_fig(plt, cm, title, path):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    _confusion_ax(ax, cm, title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
