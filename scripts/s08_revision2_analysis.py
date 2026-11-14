"""Step 8 - additional analyses of the second revision (no training required)

Computed from the result files of steps 0-7:
  * calibration (ECE, Brier score) for every protocol, model and test subset,
    including every model on the test images without a training twin
  * paired accuracy differences between each baseline/ablation and two-stage
    EfficientNet-B0 on the matched folds of seed 0 (corrected resampled t)
  * accuracy by class, image source and twin status on the official test partition
  * seed-0 estimates that go with the bootstrap confidence intervals
  * checks on the source attribution and on patient-identifier coverage
  * predictive values for tumour versus no tumour at assumed prevalences
  * summary of the grouping-sensitivity runs, if results/sensitivity/ is present

usage:
  python scripts/s08_revision2_analysis.py --results results --tables tables
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import CLASSES, bootstrap_ci, compute_metrics, corrected_t_ci, ece_score

RATIO = 0.25  # n_test / n_train in five-fold cross-validation
PROBS = ["p_glioma", "p_meningioma", "p_notumor", "p_pituitary"]


def ms(v, scale=1.0):
    v = np.asarray(v, float) * scale
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else None]


def load_cv(folder, model="efficientnet_b0", mode="two_stage"):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(folder, f"{model}__{mode}__s*_f*.json")))]
    return sorted(runs, key=lambda r: (r["seed"], r["fold"]))


def cv_summary(runs):
    out = {"runs": len(runs)}
    for k in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "ece", "brier"]:
        v = [r[k] for r in runs]
        out[k] = ms(v)
        out[k + "_ci"] = corrected_t_ci(v, RATIO)[2] if len(v) > 1 else None
    out["max_group_overlap_with_test"] = max(r["group_overlap_with_test"] for r in runs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tables", default="tables")
    a = ap.parse_args()
    R = a.results
    os.makedirs(a.tables, exist_ok=True)
    os.makedirs(os.path.join(R, "revision2"), exist_ok=True)
    out = {}

    # ---------------------------------------------------------------- image index
    g = pd.read_csv(os.path.join(R, "groups.csv"))
    g98 = pd.read_csv(os.path.join(R, "dedup_098", "groups.csv"))
    aud = pd.read_csv(os.path.join(R, "audit_index.csv"))
    g = g.merge(aud[["rel", "width", "height", "mode"]], on="rel")
    g["figshare"] = g.patient_id.notna()
    g["is512"] = (g.width == 512) & (g.height == 512)
    g["twin98"] = g98.set_index("rel").loc[g.rel, "test_has_train_twin"].values
    test = g[g.split == "test"]
    train = g[g.split == "train"]

    # ---------------------------------------------------------------- cross-validation: calibration and paired differences
    configs = {
        "EfficientNet-B0, two-stage": ("grouped", "efficientnet_b0", "two_stage"),
        "EfficientNet-B0, single-stage": ("grouped", "efficientnet_b0", "single_stage"),
        "EfficientNet-B0, frozen backbone": ("grouped", "efficientnet_b0", "frozen"),
        "MobileNetV3-Large, two-stage": ("grouped", "mobilenet_v3_large", "two_stage"),
        "ShuffleNetV2 1.0x, two-stage": ("grouped", "shufflenet_v2_x1_0", "two_stage"),
        "ResNet-18, two-stage": ("grouped", "resnet18", "two_stage"),
        "EfficientNet-B0, two-stage (image-level split)": ("naive", "efficientnet_b0", "two_stage"),
    }
    cv = {name: load_cv(os.path.join(R, "cv", proto), model, mode) for name, (proto, model, mode) in configs.items()}
    out["cv"] = {name: cv_summary(r) for name, r in cv.items()}
    ref = [r for r in cv["EfficientNet-B0, two-stage"] if r["seed"] == 0]
    out["cv_reference_seed0_accuracy"] = ms([r["accuracy"] for r in ref], 100)
    rows = []
    for name, runs in cv.items():
        s = out["cv"][name]
        row = {"model": name, "runs": s["runs"], "accuracy_mean": s["accuracy"][0] * 100, "accuracy_sd": s["accuracy"][1] * 100,
               "ece_mean": s["ece"][0], "ece_sd": s["ece"][1], "brier_mean": s["brier"][0], "brier_sd": s["brier"][1]}
        if "image-level" not in name and name != "EfficientNet-B0, two-stage":
            assert [r["fold"] for r in runs] == [0, 1, 2, 3, 4] and all(r["n_test"] == q["n_test"] for r, q in zip(runs, ref))
            d = np.array([r["accuracy"] - q["accuracy"] for r, q in zip(runs, ref)]) * 100
            k = len(d)
            se = d.std(ddof=1) * np.sqrt(1 / k + RATIO)
            t = stats.t.ppf(0.975, k - 1)
            row.update({"paired_diff_mean": d.mean(), "paired_diff_sd": d.std(ddof=1), "ci_low": d.mean() - t * se,
                        "ci_high": d.mean() + t * se, "p_corrected": 2 * stats.t.sf(abs(d.mean() / se), k - 1),
                        "folds_higher": int((d > 0).sum())})
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.tables, "table5a_calibration_and_paired_differences.csv"), index=False)
    out["paired"] = {r["model"]: {k: float(v) for k, v in r.items() if k.startswith(("paired", "ci_", "p_", "folds"))} for r in rows if "paired_diff_mean" in r}

    # ---------------------------------------------------------------- official test partition
    lab = {c: i for i, c in enumerate(CLASSES)}
    cols = ["rel", "figshare", "test_has_train_twin", "twin98", "cls"]
    P = []
    for s in (0, 1, 2):
        p = pd.read_csv(os.path.join(R, "test", f"efficientnet_b0__two_stage__s{s}__official_predictions.csv")).merge(test[cols], on="rel")
        assert len(p) == 1600 and (p.true == p.cls).all()
        P.append(p)

    def met(p):
        y = p.true.map(lab).values
        pr = p[PROBS].values
        return {"n": len(p), "correct": int((p.true == p.pred).sum()), "acc": float((p.true == p.pred).mean()) * 100,
                "ece": ece_score(y, pr), "brier": float(((pr - np.eye(4)[y]) ** 2).sum(1).mean())}

    subsets = {
        "All": lambda p: p,
        "With a training twin (NCC >= 0.95 or same patient)": lambda p: p[p.test_has_train_twin],
        "Without a training twin (NCC >= 0.95 or same patient)": lambda p: p[~p.test_has_train_twin],
        "Without a near-duplicate (NCC >= 0.98, no patient links)": lambda p: p[~p.twin98],
        "Gliomas from the figshare source": lambda p: p[(p.cls == "glioma") & p.figshare],
        "Gliomas from other sources": lambda p: p[(p.cls == "glioma") & ~p.figshare],
    }
    rows = []
    for name, f in subsets.items():
        M = [met(f(p)) for p in P]
        rows.append({"subset": name, "images": M[0]["n"], "accuracy_mean": np.mean([m["acc"] for m in M]), "accuracy_sd": np.std([m["acc"] for m in M], ddof=1),
                     "ece_mean": np.mean([m["ece"] for m in M]), "ece_sd": np.std([m["ece"] for m in M], ddof=1),
                     "brier_mean": np.mean([m["brier"] for m in M]), "brier_sd": np.std([m["brier"] for m in M], ddof=1),
                     "accuracy_seed0": M[0]["acc"], "correct_per_model": "/".join(str(m["correct"]) for m in M)})
    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.tables, "table8_calibration_by_test_subset.csv"), index=False)
    out["test_subsets"] = rows

    # table 5(b): calibration of every model on test images without a training twin
    rows = []
    models = [("EfficientNet-B0, two-stage", "efficientnet_b0", "two_stage", (0, 1, 2)), ("EfficientNet-B0, single-stage", "efficientnet_b0", "single_stage", (0,)),
              ("EfficientNet-B0, frozen backbone", "efficientnet_b0", "frozen", (0,)), ("MobileNetV3-Large, two-stage", "mobilenet_v3_large", "two_stage", (0,)),
              ("ShuffleNetV2 1.0x, two-stage", "shufflenet_v2_x1_0", "two_stage", (0,)), ("ResNet-18, two-stage", "resnet18", "two_stage", (0,))]
    for name, model, mode, seeds in models:
        M = [met(pd.read_csv(os.path.join(R, "test", f"{model}__{mode}__s{s}__clean_predictions.csv"))) for s in seeds]
        row = {"model": name, "final_models": len(M), "images": M[0]["n"]}
        for k in ["acc", "ece", "brier"]:
            row[k + "_mean"] = np.mean([m[k] for m in M])
            row[k + "_sd"] = np.std([m[k] for m in M], ddof=1) if len(M) > 1 else np.nan
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(a.tables, "table5b_no_twin_calibration.csv"), index=False)
    out["no_twin_calibration_by_model"] = rows

    # table 9: class x source x twin status
    rows = []
    for c in CLASSES + ["all"]:
        for src in (True, False):
            row = {"class": c, "source": "figshare" if src else "other"}
            for tw in (True, False):
                acc, n = [], 0
                for p in P:
                    q = p[(p.figshare == src) & (p.test_has_train_twin == tw)]
                    if c != "all":
                        q = q[q.cls == c]
                    n = len(q)
                    acc.append((q.true == q.pred).mean() * 100 if n else np.nan)
                key = "with_twin" if tw else "without_twin"
                row[key + "_n"] = n
                row[key + "_accuracy_mean"] = np.mean(acc) if n else np.nan
                row[key + "_accuracy_sd"] = np.std(acc, ddof=1) if n else np.nan
            if row["with_twin_n"] + row["without_twin_n"]:
                rows.append(row)
    pd.DataFrame(rows).round(2).to_csv(os.path.join(a.tables, "table9_accuracy_by_class_source_twin.csv"), index=False)
    out["table9"] = rows
    nt = test[~test.test_has_train_twin]
    out["no_twin_composition"] = {"n": len(nt), "other_source": int((~nt.figshare).sum()),
                                  "by_class": nt.cls.value_counts().to_dict(),
                                  "figshare_test_with_twin": int((test.figshare & test.test_has_train_twin).sum()), "figshare_test": int(test.figshare.sum())}

    # other-source gliomas: absolute counts per final model
    out["other_source_gliomas"] = []
    for s, p in enumerate(P):
        q = p[(p.cls == "glioma") & ~p.figshare]
        d = {"seed": s, "n": len(q)}
        d.update({f"predicted_{c}": int((q.pred == c).sum()) for c in CLASSES})
        d["median_confidence_no_tumour_errors"] = float(q[q.pred == "notumor"].confidence.median())
        out["other_source_gliomas"].append(d)

    # seed-0 estimates that accompany the bootstrap intervals
    p = P[0]
    y, pr = p.true.map(lab).values, p[PROBS].values
    m = compute_metrics(y, pr)
    out["seed0_official"] = {k: float(m[k]) for k in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "ece", "brier"]}
    out["seed0_official_bootstrap_ci"] = bootstrap_ci(y, pr)
    q = (~p.test_has_train_twin).values
    ok, yy = (p.true == p.pred).values[q], y[q]
    rng = np.random.default_rng(0)
    boot = [ok[np.concatenate([rng.choice(np.where(yy == c)[0], (yy == c).sum()) for c in np.unique(yy)])].mean() for _ in range(2000)]
    out["seed0_no_twin"] = {"accuracy": float(ok.mean()), "bootstrap_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}

    # ---------------------------------------------------------------- source attribution and patient coverage
    gl = g[g.cls == "glioma"]
    other_gl = test[(test.cls == "glioma") & ~test.figshare]
    other_tumour = g[~g.figshare & (g.cls != "notumor")]
    out["source_checks"] = {
        "matched_gliomas": int(gl.figshare.sum()), "matched_gliomas_512": int((gl.figshare & gl.is512).sum()),
        "other_source_test_gliomas": len(other_gl), "other_source_test_gliomas_512": int(other_gl.is512.sum()),
        "other_source_test_glioma_distinct_sizes": int(other_gl.groupby(["width", "height"]).ngroups),
        "other_source_test_glioma_modes": other_gl["mode"].value_counts().to_dict(),
        "other_source_tumour_images": len(other_tumour),
        "other_source_tumour_images_512": other_tumour[other_tumour.is512].cls.value_counts().to_dict(),
    }
    no_id = train[~train.figshare]
    out["patient_coverage"] = {
        "images_with_id": int(g.figshare.sum()), "images": len(g), "train_with_id": int(train.figshare.sum()), "train": len(train),
        "train_without_id_by_class": no_id.cls.value_counts().to_dict(),
        "train_without_id_in_multi_image_group": int((no_id.group_size > 1).sum()),
        "train_gliomas_with_id": int((train[train.cls == "glioma"].figshare).sum()),
    }

    # ---------------------------------------------------------------- predictive values (tumour versus no tumour)
    ni = CLASSES.index("notumor")
    sens, spec = [], []
    for r in cv["EfficientNet-B0, two-stage"]:
        c = np.array(r["confusion"])
        tum = np.delete(c, ni, axis=0)
        sens.append(1 - tum[:, ni].sum() / tum.sum())
        spec.append(c[ni, ni] / c[ni].sum())
    se, sp = float(np.mean(sens)), float(np.mean(spec))
    rows = []
    for prev in [0.75, 0.50, 0.20, 0.10, 0.05, 0.02]:
        rows.append({"prevalence": prev, "ppv": se * prev / (se * prev + (1 - sp) * (1 - prev)) * 100,
                     "npv": sp * (1 - prev) / (sp * (1 - prev) + (1 - se) * prev) * 100})
    pd.DataFrame(rows).round(2).to_csv(os.path.join(a.tables, "table10_predictive_values.csv"), index=False)
    out["predictive_values"] = {"sensitivity": ms(sens, 100), "specificity": ms(spec, 100), "rows": rows}

    # ---------------------------------------------------------------- grouping-sensitivity runs (optional)
    sens_dir = os.path.join(R, "sensitivity")
    if os.path.isdir(sens_dir):
        labels = {"ncc098_patients": "Grouped: NCC >= 0.98 + patients", "ncc095_duponly": "Grouped: NCC >= 0.95, no patient identifiers",
                  "ncc098_duponly": "Grouped: NCC >= 0.98, no patient identifiers"}
        rows = [dict(protocol="Grouped: NCC >= 0.95 + patients (main analysis)", **out["cv"]["EfficientNet-B0, two-stage"])]
        out["sensitivity"] = {}
        for key, label in labels.items():
            runs = load_cv(os.path.join(sens_dir, key, "cv", "grouped"))
            if not runs:
                continue
            out["sensitivity"][key] = cv_summary(runs)
            out["sensitivity"][key]["n_train_groups"] = int(pd.read_csv(os.path.join(sens_dir, f"groups_{key}.csv")).query("split == 'train'").group_id.nunique())
            rows.append(dict(protocol=label, **out["sensitivity"][key]))
        rows.append(dict(protocol="Image level (conventional)", **out["cv"]["EfficientNet-B0, two-stage (image-level split)"]))
        flat = []
        for r in rows:
            f = {"protocol": r["protocol"], "runs": r["runs"]}
            for k in ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "ece", "brier"]:
                f[k + "_mean"], f[k + "_sd"] = r[k]
                if r.get(k + "_ci"):
                    f[k + "_ci_low"], f[k + "_ci_high"] = r[k + "_ci"]
            flat.append(f)
        pd.DataFrame(flat).round(4).to_csv(os.path.join(a.tables, "table3_cross_validation_grouping_sensitivity.csv"), index=False)

    json.dump(out, open(os.path.join(R, "revision2", "revision2_values.json"), "w"), indent=1, default=float)
    print("written", os.path.join(R, "revision2", "revision2_values.json"))


if __name__ == "__main__":
    main()
