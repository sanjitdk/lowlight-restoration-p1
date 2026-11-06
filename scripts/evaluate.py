"""Evaluate all methods + ablations and write results tables.

Usage:
  python scripts/evaluate.py --split val                        # development
  python scripts/evaluate.py --split test --config configs/final.yaml   # FINAL, run once
  python scripts/evaluate.py --split unseen                     # no-reference only (NIQE etc.)

Splits: val (50 LOL-v1 our485 images), test (= test_lol + test_lolv2),
        test_lol (LOL-v1 eval15), test_lolv2 (LOL-v2 Real test 100), unseen (data/unseen/*).

Integrity: the first test run writes results/test_lock.json with the config's hash.
A later test run with a different config is refused (no tuning on test).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics as M  # noqa: E402
from src.dl.zerodce import load_model  # noqa: E402
from src.hybrid import HybridEnhancer, load_config  # noqa: E402
from src.io_utils import load_image, save_image  # noqa: E402

# (name, mode, kwargs, which model)
METHODS = [
    ("input (no enhancement)", None, {}, None),
    ("classical (fixed params)", "classical", {"adaptive": False}, "ft"),
    ("classical (adaptive)", "classical", {}, "ft"),
    ("DL: Zero-DCE++ pretrained", "dl", {}, "pre"),
    ("DL: Zero-DCE++ fine-tuned", "dl", {}, "ft"),
    ("hybrid (fixed params)", "hybrid", {"adaptive": False}, "ft"),
    ("hybrid (adaptive) [ours]", "hybrid", {}, "ft"),
    ("auto (quality selector)", "auto", {}, "ft"),
]
ABLATIONS = [
    ("hybrid - illumination guidance", "hybrid", {"disable": ("guidance",)}, "ft"),
    ("hybrid - denoising", "hybrid", {"disable": ("denoise",)}, "ft"),
    ("hybrid - fusion (DL branch only)", "hybrid", {"disable": ("fusion",)}, "ft"),
    ("hybrid - CLAHE", "hybrid", {"disable": ("clahe",)}, "ft"),
    ("hybrid - colour correction", "hybrid", {"disable": ("color",)}, "ft"),
    ("hybrid - Retinex (plain gamma)", "hybrid", {"disable": ("retinex",)}, "ft"),
]


def split_items(split):
    sp = ROOT / "data/splits"
    if split == "val":
        b = ROOT / "data/LOL/our485"
        return [(b / "low" / n, b / "high" / n, "val") for n in (sp / "val.txt").read_text().split()]
    if split == "test_lol":
        b = ROOT / "data/LOL/eval15"
        return [(b / "low" / n, b / "high" / n, "LOL-v1 eval15") for n in (sp / "test_lol.txt").read_text().split()]
    if split == "test_lolv2":
        b = ROOT / "data/LOLv2/Real_captured/Test"
        return [(b / "Input" / n, b / "GT" / n, "LOL-v2 Real test") for n in (sp / "test_lolv2.txt").read_text().split()]
    if split == "test":
        return split_items("test_lol") + split_items("test_lolv2")
    if split == "unseen":
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        return [(p, None, "unseen") for p in sorted((ROOT / "data/unseen").iterdir()) if p.suffix.lower() in exts]
    raise SystemExit(f"unknown split {split}")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def check_test_lock(split, cfg_path):
    if not split.startswith("test"):
        return
    h = hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest()[:16]
    w = hashlib.sha256(Path(load_config(cfg_path)["weights"]).read_bytes()).hexdigest()[:16]
    lock = ROOT / "results/test_lock.json"
    if lock.exists():
        prev = json.loads(lock.read_text())
        if prev["config_sha"] != h or prev["weights_sha"] != w:
            raise SystemExit("Refusing: test set was already evaluated with a different config/weights. "
                             "Test results must come from the frozen config (see results/test_lock.json).")
    else:
        lock.write_text(json.dumps({"config": str(cfg_path), "config_sha": h, "weights_sha": w,
                                    "git": git_hash(), "time": time.strftime("%Y-%m-%d %H:%M")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--pretrained", default="weights/zerodcepp_pretrained.pth")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--no-niqe", action="store_true")
    ap.add_argument("--save-images", type=int, default=6, help="save outputs for first N images")
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    args = ap.parse_args()
    if args.threads:
        import torch
        torch.set_num_threads(args.threads)

    cfg_path = ROOT / args.config
    check_test_lock(args.split, cfg_path)
    cfg = load_config(str(cfg_path))
    H_ft = HybridEnhancer(cfg)
    H_pre = HybridEnhancer(cfg, model=load_model(args.pretrained))
    methods = METHODS + ([] if args.no_ablation else ABLATIONS)
    items = split_items(args.split)[: args.max_n]
    out_dir = ROOT / "results" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    # warm-up so the first image's latency isn't inflated by lazy init
    H_ft.enhance(load_image(items[0][0]), mode="hybrid")

    rows = []
    for i, (lp, gp, subset) in enumerate(items):
        lo = load_image(lp, None)
        gt = load_image(gp, None) if gp else None
        for name, mode, kw, which in methods:
            H = H_pre if which == "pre" else H_ft
            r = {"image": lp.name, "subset": subset, "method": name}
            with M.timer(r):
                out = lo if mode is None else H.enhance(lo, mode=mode, **kw)
            if mode is None:
                r["latency_ms"] = 0.0
            if gt is not None:
                r.update(M.full_reference(gt, out))
            else:
                r["noise_sigma"] = M.noise_sigma(out)
            if not args.no_niqe and (name, mode, kw, which) in METHODS:  # NIQE for main methods only
                r["niqe"] = M.niqe(out)
            rows.append(r)
            if i < args.save_images:
                safe = name.split(" [")[0].replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
                save_image(out, out_dir / "images" / f"{lp.stem}__{safe}.png")
        print(f"[{i + 1}/{len(items)}] {lp.name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_image.csv", index=False)
    cols = [c for c in ["psnr", "ssim", "lpips", "niqe", "latency_ms", "noise_sigma", "edge_corr", "edge_f1"]
            if c in df.columns]
    order = [m[0] for m in methods]
    summary = df.groupby(["subset", "method"])[cols].mean().reset_index()
    summary["method"] = pd.Categorical(summary["method"], order, ordered=True)
    summary = summary.sort_values(["subset", "method"])
    summary.to_csv(out_dir / "summary.csv", index=False)
    meta = {"split": args.split, "config": args.config, "git": git_hash(), "n_images": len(items),
            "time": time.strftime("%Y-%m-%d %H:%M"), "cpu_threads": __import__("torch").get_num_threads()}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
