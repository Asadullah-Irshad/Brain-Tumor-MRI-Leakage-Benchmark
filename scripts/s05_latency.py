"""Step 5 - measured efficiency.

Reports, for each model: parameters, fp32 weight-file size, GFLOPs (multiply-
adds x2) at 224x224, and MEASURED wall-clock CPU latency (median and 95th
percentile over --runs single-image forward passes after --warmup passes),
plus per-image time at batch 32. PyTorch eager and, if installed, ONNX Runtime.
The CPU model, thread count and library versions are recorded.

Run this on the CPU you want to report (e.g. your laptop), not on Colab.
usage: python scripts/s05_latency.py --out results [--weights results/test/efficientnet_b0__two_stage__s0.pt]
"""
import argparse
import os
import tempfile
import time

import numpy as np

from common import MODEL_NAMES, Config, build_model, count_params, environment_info, save_json


def bench(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES)
    ap.add_argument("--weights", default=None, help="optional trained EfficientNet-B0 weights")
    ap.add_argument("--threads", type=int, default=0, help="0 = PyTorch default")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = Config()
    cfg.pretrained = False
    size = 224
    if a.smoke:
        a.warmup, a.runs = 2, 5
    res = {"env": environment_info(), "torch_threads": torch.get_num_threads(), "input": [1, 3, size, size],
           "precision": "fp32", "device": "cpu", "models": {}}
    for name in a.models:
        model, _ = build_model(name, cfg)
        if name == "efficientnet_b0" and a.weights and os.path.exists(a.weights):
            model.load_state_dict(torch.load(a.weights, map_location="cpu"))
        model.eval()
        r = {"params": count_params(model)}
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "w.pt")
            torch.save(model.state_dict(), f)
            r["weights_mb"] = os.path.getsize(f) / 1e6
        x1 = torch.randn(1, 3, size, size)
        x32 = torch.randn(32, 3, size, size)
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                with torch.no_grad():
                    model(x1)
            r["gflops"] = fc.get_total_flops() / 1e9
        except Exception:
            r["gflops"] = None
        with torch.inference_mode():
            r["pytorch_ms_median"], r["pytorch_ms_p95"] = bench(lambda: model(x1), a.warmup, a.runs)
            med32, _ = bench(lambda: model(x32), max(2, a.warmup // 5), max(5, a.runs // 10))
            r["pytorch_batch32_ms_per_image"] = med32 / 32
        try:
            import onnxruntime as ort
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "m.onnx")
                try:  # legacy exporter first (no extra dependency), then the newer one
                    torch.onnx.export(model, x1, f, input_names=["x"], output_names=["y"],
                                      opset_version=17, dynamo=False)
                except Exception:
                    torch.onnx.export(model, x1, f, input_names=["x"], output_names=["y"])
                so = ort.SessionOptions()
                if a.threads:
                    so.intra_op_num_threads = a.threads
                sess = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
                xn = x1.numpy()
                r["onnxruntime_ms_median"], r["onnxruntime_ms_p95"] = bench(
                    lambda: sess.run(None, {"x": xn}), a.warmup, a.runs)
                r["onnxruntime_version"] = ort.__version__
        except Exception as e:
            r["onnxruntime_ms_median"] = None
            r["onnxruntime_error"] = str(e)[:200]
        res["models"][name] = r
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    os.makedirs(a.out, exist_ok=True)
    save_json(res, os.path.join(a.out, "latency.json"))
    print("wrote", os.path.join(a.out, "latency.json"))


if __name__ == "__main__":
    main()
