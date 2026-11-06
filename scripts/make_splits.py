"""Create the fixed train/val split from LOL-v1 our485 and list the test sets.

Usage: python scripts/make_splits.py [--data data] [--val 50] [--seed 42]

Writes data/splits/{train,val,test_lol,test_lolv2}.txt with one image filename per line.
Val is used for all tuning; test lists are only used by the final evaluation run.
"""
import argparse
import random
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def find_dir(root: Path, *candidates):
    """Return the first existing candidate path (dataset zips use different casing)."""
    for c in candidates:
        p = root / c
        if p.is_dir():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data)
    out = root / "splits"
    out.mkdir(parents=True, exist_ok=True)

    train_low = find_dir(root, "LOL/our485/low", "LOL/train/low")
    if train_low is None:
        raise SystemExit("LOL-v1 our485/low not found under data/LOL. Download the dataset first.")
    names = list_images(train_low)
    random.Random(args.seed).shuffle(names)
    val, train = sorted(names[: args.val]), sorted(names[args.val:])
    (out / "train.txt").write_text("\n".join(train) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    print(f"train={len(train)} val={len(val)} (seed={args.seed})")

    test_lol = find_dir(root, "LOL/eval15/low", "LOL/test/low")
    if test_lol:
        t = list_images(test_lol)
        (out / "test_lol.txt").write_text("\n".join(t) + "\n")
        print(f"test_lol={len(t)}")

    test_v2 = find_dir(root, "LOLv2/Real_captured/Test/Input", "LOLv2/Real_captured/Test/Low",
                       "LOLv2/Real_captured/Test/low")
    if test_v2:
        t = list_images(test_v2)
        (out / "test_lolv2.txt").write_text("\n".join(t) + "\n")
        print(f"test_lolv2={len(t)}")


if __name__ == "__main__":
    main()
