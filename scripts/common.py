"""Shared utilities for the brain-tumour MRI leakage benchmark.

Every script in this kit imports from here so that data loading, models,
training, metrics and configuration are identical across experiments.
"""
import hashlib
import json
import os
import platform
import random
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABEL = {c: i for i, c in enumerate(CLASSES)}
CLASS_PRETTY = {"glioma": "Glioma", "meningioma": "Meningioma",
                "notumor": "No tumour", "pituitary": "Pituitary"}
_ALIASES = {
    "glioma": "glioma", "glioma_tumor": "glioma", "gliomas": "glioma",
    "meningioma": "meningioma", "meningioma_tumor": "meningioma",
    "meningiomas": "meningioma",
    "notumor": "notumor", "no_tumor": "notumor", "no_tumour": "notumor",
    "notumour": "notumor", "normal": "notumor", "healthy": "notumor",
    "no": "notumor",
    "pituitary": "pituitary", "pituitary_tumor": "pituitary",
    "pituitary_tumour": "pituitary", "pituitary_macroadenoma": "pituitary",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# configuration (the values reported in the paper)
# --------------------------------------------------------------------------
@dataclass
class Config:
    img_size: int = 224
    batch_size: int = 32
    head_epochs: int = 15        # stage 1: classifier head only
    head_lr: float = 1e-3
    ft_epochs: int = 30          # stage 2: whole network
    ft_lr: float = 5e-5
    single_stage_lr: float = 1e-4  # ablation: no head warm-up
    patience: int = 8            # early stopping on validation loss
    dropout: float = 0.2
    val_fraction: float = 0.15   # group-aware, taken from the training folds only
    # augmentation (training only)
    hflip_p: float = 0.5
    rotation_deg: float = 10.0
    crop_scale_min: float = 0.90   # random-resized-crop 'zoom'
    brightness: float = 0.10
    contrast: float = 0.10
    num_workers: int = 2
    amp: bool = True
    pretrained: bool = True

    def smoke(self):
        """Tiny settings used only to check the pipeline runs end to end."""
        self.img_size = 64
        self.batch_size = 8
        self.head_epochs = 1
        self.ft_epochs = 1
        self.patience = 1
        self.num_workers = 0
        self.pretrained = False
        return self


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_info():
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count()}
    try:
        import torch
        import torchvision
        info["torch"] = torch.__version__
        info["torchvision"] = torchvision.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:  # pragma: no cover
        pass
    info["cpu_model"] = cpu_model_name()
    return info


def cpu_model_name():
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode().split("\n")
            return [l.strip() for l in out if l.strip() and l.strip() != "Name"][0]
    except Exception:
        pass
    return platform.processor() or "unknown"


# --------------------------------------------------------------------------
# data indexing
# --------------------------------------------------------------------------
def _norm_class(name):
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(key)


def index_class_folders(root, split_name):
    """Index <root>/<class>/<image> into a DataFrame."""
    rows = []
    for cdir in sorted(os.listdir(root)):
        full = os.path.join(root, cdir)
        if not os.path.isdir(full):
            continue
        cls = _norm_class(cdir)
        if cls is None:
            print(f"  [warn] skipping unrecognised class folder: {full}")
            continue
        for dirpath, _, files in os.walk(full):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in IMG_EXT:
                    rows.append({"path": os.path.join(dirpath, f), "split": split_name,
                                 "cls": cls, "label": CLASS_LABEL[cls]})
    return pd.DataFrame(rows)


def find_subdir(root, names):
    for d in os.listdir(root):
        if d.lower() in names and os.path.isdir(os.path.join(root, d)):
            return os.path.join(root, d)
    raise FileNotFoundError(f"none of {names} found under {root}")


def index_nickparvar(root):
    """Index the Kaggle Brain Tumor MRI Dataset (Training/ and Testing/ folders)."""
    tr = index_class_folders(find_subdir(root, {"training", "train"}), "train")
    te = index_class_folders(find_subdir(root, {"testing", "test"}), "test")
    df = pd.concat([tr, te], ignore_index=True)
    df["rel"] = [os.path.relpath(p, root).replace("\\", "/") for p in df["path"]]
    return df


def index_external(root):
    """External dataset: either <root>/<class>/... or <root>/<split>/<class>/...

    If split folders exist, only a folder named test/testing is used (so the
    external images are never ones a dataset curator meant for training)."""
    subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if any(_norm_class(d) for d in subs):
        df = index_class_folders(root, "external")
    else:
        df = index_class_folders(find_subdir(root, {"testing", "test"}), "external")
    df["rel"] = [os.path.relpath(p, root).replace("\\", "/") for p in df["path"]]
    return df


# --------------------------------------------------------------------------
# image cache (decode each image once; big speed-up on Colab/Kaggle)
# --------------------------------------------------------------------------
def build_cache(paths, size, cache_file):
    """Returns uint8 array (N, size, size, 3). Cached as .npy keyed by file list."""
    key = hashlib.md5(("|".join(paths) + f"@{size}").encode()).hexdigest()[:12]
    cache_file = f"{cache_file}_{size}_{key}.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file, mmap_mode="r")
    os.makedirs(os.path.dirname(os.path.abspath(cache_file)), exist_ok=True)
    arr = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    t0 = time.time()
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
        if (i + 1) % 1000 == 0:
            print(f"  cached {i + 1}/{len(paths)} images ({time.time() - t0:.0f}s)")
    np.save(cache_file, arr)
    return arr


def get_torch_dataset(images, labels, cfg, train):
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if train:
        aug = T.Compose([
            T.RandomHorizontalFlip(cfg.hflip_p),
            T.RandomRotation(cfg.rotation_deg),
            T.RandomResizedCrop(cfg.img_size, scale=(cfg.crop_scale_min, 1.0),
                                ratio=(0.95, 1.05), antialias=True),
            T.ColorJitter(brightness=cfg.brightness, contrast=cfg.contrast),
        ])
    else:
        aug = None

    class _DS(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            x = torch.from_numpy(np.ascontiguousarray(images[i])).permute(2, 0, 1)  # uint8 CHW
            if aug is not None:
                x = aug(x)
            x = x.float().div(255.0)
            x = (x - mean) / std
            return x, int(labels[i])

    return _DS()


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
MODEL_NAMES = ["efficientnet_b0", "resnet18", "mobilenet_v3_large", "shufflenet_v2_x1_0"]
MODEL_PRETTY = {"efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet-18",
                "mobilenet_v3_large": "MobileNetV3-Large",
                "shufflenet_v2_x1_0": "ShuffleNetV2 (1.0x)"}
MODE_PRETTY = {"two_stage": "two-stage fine-tuning", "frozen": "frozen backbone (head only)",
               "single_stage": "single-stage fine-tuning"}


def build_model(name, cfg, n_classes=4):
    """ImageNet-pretrained backbone; global average pooling (built in);
    dropout + linear softmax head. Returns (model, head_module)."""
    import torch.nn as nn
    import torchvision.models as tvm
    pre = cfg.pretrained
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pre else None)
        m.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(1280, n_classes))
        head = m.classifier
    elif name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pre else None)
        m.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(512, n_classes))
        head = m.fc
    elif name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pre else None)
        m.classifier[2].p = cfg.dropout
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        head = m.classifier
    elif name == "shufflenet_v2_x1_0":
        m = tvm.shufflenet_v2_x1_0(weights=tvm.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pre else None)
        m.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(1024, n_classes))
        head = m.fc
    else:
        raise ValueError(name)
    return m, head


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def _loader(ds, cfg, shuffle):
    from torch.utils.data import DataLoader
    import torch
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False, persistent_workers=cfg.num_workers > 0)


def _run_epoch(model, loader, crit, dev, opt=None, freeze_backbone=False, head=None, cfg=None):
    import torch
    train = opt is not None
    if train:
        if freeze_backbone:
            model.eval()          # keep BN statistics of the frozen backbone fixed
            head.train()
        else:
            model.train()
    else:
        model.eval()
    use_amp = bool(cfg and cfg.amp and dev.type == "cuda")
    scaler = _run_epoch.scaler
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                out = model(x)
                loss = crit(out, y)
            if train:
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()
            tot += loss.item() * len(y)
            n += len(y)
    return tot / max(n, 1)


_run_epoch.scaler = None


def fit(model, head, tr_imgs, tr_y, va_imgs, va_y, mode, cfg, dev, verbose=True):
    """Train with early stopping on validation loss. Returns history dict."""
    import copy
    import torch
    import torch.nn as nn
    _run_epoch.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and dev.type == "cuda"))
    tr_ds = get_torch_dataset(tr_imgs, tr_y, cfg, train=True)
    va_ds = get_torch_dataset(va_imgs, va_y, cfg, train=False)
    tr_dl, va_dl = _loader(tr_ds, cfg, True), _loader(va_ds, cfg, False)
    crit = nn.CrossEntropyLoss()
    model.to(dev)

    if mode == "two_stage":
        stages = [("head", cfg.head_epochs, cfg.head_lr), ("full", cfg.ft_epochs, cfg.ft_lr)]
    elif mode == "frozen":
        stages = [("head", cfg.head_epochs + cfg.ft_epochs, cfg.head_lr)]
    elif mode == "single_stage":
        stages = [("full", cfg.head_epochs + cfg.ft_epochs, cfg.single_stage_lr)]
    else:
        raise ValueError(mode)

    hist = {"stages": []}
    for sname, epochs, lr in stages:
        frozen = sname == "head"
        for p in model.parameters():
            p.requires_grad = not frozen
        for p in head.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        best, best_state, best_ep, wait = float("inf"), None, 0, 0
        log = []
        for ep in range(1, epochs + 1):
            t0 = time.time()
            trl = _run_epoch(model, tr_dl, crit, dev, opt, frozen, head, cfg)
            val = _run_epoch(model, va_dl, crit, dev, None, cfg=cfg)
            log.append({"epoch": ep, "train_loss": trl, "val_loss": val})
            if verbose:
                print(f"    [{sname}] ep {ep:02d} train {trl:.4f} val {val:.4f} ({time.time()-t0:.0f}s)")
            if val < best - 1e-4:
                best, best_ep, wait = val, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        hist["stages"].append({"stage": sname, "lr": lr, "best_epoch": best_ep,
                               "best_val_loss": best, "epochs_run": len(log), "log": log})
    return hist


def predict(model, imgs, cfg, dev):
    import torch
    ds = get_torch_dataset(imgs, np.zeros(len(imgs), dtype=int), cfg, train=False)
    dl = _loader(ds, cfg, False)
    model.eval().to(dev)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast(device_type=dev.type, enabled=bool(cfg.amp and dev.type == "cuda")):
                out.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
    return np.concatenate(out)


def group_val_split(groups, labels, frac, seed):
    """Group-aware, approximately stratified validation split (no group crosses)."""
    from sklearn.model_selection import StratifiedGroupKFold
    k = max(2, int(round(1.0 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(np.zeros(len(labels)), labels, groups))
    return tr, va


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def ece_score(y, p, n_bins=15):
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y, p):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, roc_auc_score,
                                 confusion_matrix)
    y = np.asarray(y)
    pred = p.argmax(1)
    labels = list(range(len(CLASSES)))
    P, R, F, S = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(P.mean()), "macro_recall": float(R.mean()),
        "macro_f1": float(F.mean()),
        "ece": ece_score(y, p),
        "brier": float(np.mean(np.sum((p - np.eye(len(CLASSES))[y]) ** 2, 1))),
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
        "per_class": {},
    }
    try:
        out["macro_auc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        out["macro_auc"] = float("nan")
    for k, c in enumerate(CLASSES):
        try:
            auc = float(roc_auc_score((y == k).astype(int), p[:, k]))
        except ValueError:
            auc = float("nan")
        out["per_class"][c] = {"precision": float(P[k]), "recall": float(R[k]),
                               "f1": float(F[k]), "support": int(S[k]), "auc": auc}
    return out


def bootstrap_ci(y, p, n_boot=2000, seed=0, stratified=True):
    """Percentile 95% CI by resampling TEST IMAGES (class-stratified)."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx_by_c = [np.where(y == k)[0] for k in range(len(CLASSES))]
    stats = {"accuracy": [], "balanced_accuracy": [], "macro_f1": [], "macro_auc": []}
    for _ in range(n_boot):
        if stratified:
            idx = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_c if len(ix)])
        else:
            idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], p[idx]
        pr = pp.argmax(1)
        stats["accuracy"].append(accuracy_score(yy, pr))
        stats["balanced_accuracy"].append(balanced_accuracy_score(yy, pr))
        stats["macro_f1"].append(f1_score(yy, pr, average="macro", labels=list(range(len(CLASSES))), zero_division=0))
        try:
            stats["macro_auc"].append(roc_auc_score(yy, pp, multi_class="ovr", average="macro",
                                                    labels=list(range(len(CLASSES)))))
        except ValueError:
            pass
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2
            for k, v in stats.items()}


def corrected_t_ci(values, test_train_ratio, conf=0.95):
    """Mean and 95% CI over repeated-CV runs with the Nadeau-Bengio correction
    for overlapping training sets (variance multiplied by 1/J + n_test/n_train)."""
    from scipy import stats
    v = np.asarray(values, dtype=float)
    J = len(v)
    m, s2 = v.mean(), v.var(ddof=1) if J > 1 else 0.0
    se = np.sqrt((1.0 / J + test_train_ratio) * s2)
    t = stats.t.ppf(0.5 + conf / 2, J - 1) if J > 1 else float("nan")
    return float(m), float(np.sqrt(s2)), [float(m - t * se), float(m + t * se)]
