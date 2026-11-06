"""Fine-tune Zero-DCE++ on the LOL-v1 train split; select the checkpoint on val.

Run (Kaggle GPU or local):
  python -m src.dl.train --lol data/LOL --splits data/splits \
      --init weights/zerodcepp_pretrained.pth --out weights/zerodcepp_finetuned.pth

Loss = Zero-DCE++ zero-reference losses (spatial consistency, exposure, colour
constancy, illumination smoothness) + optional supervised term (L1 + SSIM to the
LOL ground truth) weighted by --sup. LOL provides paired data, so using it is
"fine-tuning with suitable training data"; --sup 0 gives the original unsupervised
recipe. Only train.txt images are used for gradients; val.txt picks the epoch.
Test images are never touched here.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .zerodce import ZeroDCEpp, load_model


# ---------------- zero-reference losses (official formulation) ----------------

class L_color(nn.Module):
    def forward(self, x):
        m = x.mean(dim=(2, 3))
        r, g, b = m[:, 0], m[:, 1], m[:, 2]
        drg, drb, dgb = (r - g) ** 2, (r - b) ** 2, (b - g) ** 2
        return torch.sqrt(drg ** 2 + drb ** 2 + dgb ** 2 + 1e-8).mean()


class L_spa(nn.Module):
    def __init__(self):
        super().__init__()
        k = {"l": [[0, 0, 0], [-1, 1, 0], [0, 0, 0]], "r": [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
             "u": [[0, -1, 0], [0, 1, 0], [0, 0, 0]], "d": [[0, 0, 0], [0, 1, 0], [0, -1, 0]]}
        for n, v in k.items():
            self.register_buffer(n, torch.tensor(v, dtype=torch.float32)[None, None])
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enh):
        o = self.pool(org.mean(1, keepdim=True))
        e = self.pool(enh.mean(1, keepdim=True))
        loss = 0
        for k in (self.l, self.r, self.u, self.d):
            loss = loss + (F.conv2d(o, k, padding=1) - F.conv2d(e, k, padding=1)) ** 2
        return loss.mean()


class L_exp(nn.Module):
    def __init__(self, patch=16, mean_val=0.6):
        super().__init__()
        self.pool = nn.AvgPool2d(patch)
        self.mean_val = mean_val

    def forward(self, x):
        return ((self.pool(x.mean(1, keepdim=True)) - self.mean_val) ** 2).mean()


def L_tv(A):
    n, _, h, w = A.shape
    dh = ((A[:, :, 1:, :] - A[:, :, :-1, :]) ** 2).sum() / ((h - 1) * w)
    dw = ((A[:, :, :, 1:] - A[:, :, :, :-1]) ** 2).sum() / (h * (w - 1))
    return 2 * (dh + dw) / n


def ssim_loss(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    mx, my = F.avg_pool2d(x, 7, 1, 3), F.avg_pool2d(y, 7, 1, 3)
    vx = F.avg_pool2d(x * x, 7, 1, 3) - mx ** 2
    vy = F.avg_pool2d(y * y, 7, 1, 3) - my ** 2
    cxy = F.avg_pool2d(x * y, 7, 1, 3) - mx * my
    s = ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    return 1 - s.mean()


# ---------------- data ----------------

def _read(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


class LOLPairs(Dataset):
    def __init__(self, lol_root, names, crop=256, train=True):
        root = Path(lol_root)
        base = root / "our485" if (root / "our485").is_dir() else root / "train"
        self.low = [base / "low" / n for n in names]
        self.high = [base / "high" / n for n in names]
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.low)

    def __getitem__(self, i):
        lo, hi = _read(self.low[i]), _read(self.high[i])
        if self.train:
            h, w = lo.shape[:2]
            y, x = random.randint(0, h - self.crop), random.randint(0, w - self.crop)
            lo, hi = lo[y:y + self.crop, x:x + self.crop], hi[y:y + self.crop, x:x + self.crop]
            if random.random() < 0.5:
                lo, hi = lo[:, ::-1], hi[:, ::-1]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return t(lo), t(hi)


@torch.no_grad()
def evaluate_psnr(model, loader, device):
    model.eval()
    vals = []
    for lo, hi in loader:
        out, _ = model(lo.to(device))
        mse = F.mse_loss(out.clamp(0, 1), hi.to(device)).item()
        vals.append(10 * np.log10(1.0 / max(mse, 1e-10)))
    model.train()
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lol", default="data/LOL")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--init", default="weights/zerodcepp_pretrained.pth")
    ap.add_argument("--out", default="weights/zerodcepp_finetuned.pth")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--sup", type=float, default=1.0, help="weight of supervised L1+SSIM term (0 = zero-reference only)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sp = Path(args.splits)
    train_names = sp.joinpath("train.txt").read_text().split()
    val_names = sp.joinpath("val.txt").read_text().split()
    assert not set(train_names) & set(val_names), "train/val overlap"

    model = load_model(args.init, device=device) if args.init else ZeroDCEpp().to(device)
    model.train()
    tl = DataLoader(LOLPairs(args.lol, train_names, args.crop, True), batch_size=args.bs, shuffle=True,
                    num_workers=2, drop_last=True)
    vl = DataLoader(LOLPairs(args.lol, val_names, train=False), batch_size=1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lcol, lspa, lexp = L_color().to(device), L_spa().to(device), L_exp().to(device)

    best = evaluate_psnr(model, vl, device)
    history = [{"epoch": 0, "val_psnr": best}]
    print(f"epoch 0 (init) val PSNR {best:.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    for ep in range(1, args.epochs + 1):
        t0, tot = time.time(), 0.0
        for lo, hi in tl:
            lo, hi = lo.to(device), hi.to(device)
            out, A = model(lo)
            loss = 1600 * L_tv(A) + lspa(lo, out) + 5 * lcol(out) + 10 * lexp(out)
            if args.sup > 0:
                loss = loss + args.sup * (F.l1_loss(out, hi) + ssim_loss(out, hi))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); tot += loss.item()
        v = evaluate_psnr(model, vl, device)
        history.append({"epoch": ep, "train_loss": tot / len(tl), "val_psnr": v})
        tag = ""
        if v > best:
            best, tag = v, "  <- best, saved"
            torch.save(model.state_dict(), args.out)
        print(f"epoch {ep} loss {tot / len(tl):.4f} val PSNR {v:.2f} ({time.time() - t0:.0f}s){tag}")
    Path(args.out).with_suffix(".history.json").write_text(json.dumps(history, indent=2))
    print(f"best val PSNR {best:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
