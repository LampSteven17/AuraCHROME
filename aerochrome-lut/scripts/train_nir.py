#!/usr/bin/env python3
"""
Train the compact RGB->NIR U-Net on paired RGB/NIR images (default: the EPFL
RGB-NIR Scene dataset). Random crops + flips, L1 loss, AMP. Saves the best
validation checkpoint to models/nir_unet.pt — the file the engine's neural path
gates on.

Usage (inside `poetry install --with neural`):
    python scripts/train_nir.py --data data/nirscene1 --epochs 300

Dataset layout: paired files named <stem>_rgb.<ext> and <stem>_nir.<ext>
(EPFL nirscene1 uses exactly this). NIR may be stored as 1- or 3-channel; we
take its first channel.
"""
import argparse
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aerochrome.neural.unet import CompactUNet


def _load(path):
    try:
        import tifffile
        if os.path.splitext(path)[1].lower() in (".tif", ".tiff"):
            return np.asarray(tifffile.imread(path))
    except Exception:
        pass
    from PIL import Image
    return np.asarray(Image.open(path))


def _norm(a):
    a = np.asarray(a).astype(np.float32)
    return a / (65535.0 if a.dtype == np.uint16 or a.max() > 255 else 255.0)


class PairDataset(Dataset):
    def __init__(self, pairs, crop=256, train=True, photometric=True):
        self.pairs, self.crop, self.train = pairs, crop, train
        self.photometric = photometric and train

    def __len__(self):
        return len(self.pairs) * (8 if self.train else 1)

    def __getitem__(self, i):
        rgb_p, nir_p = self.pairs[i % len(self.pairs)]
        rgb = _norm(_load(rgb_p))[..., :3]
        nir = _norm(_load(nir_p))
        if nir.ndim == 3:
            nir = nir[..., 0]
        h, w = rgb.shape[:2]
        c = self.crop
        if self.train and h >= c and w >= c:
            y, x = random.randint(0, h - c), random.randint(0, w - c)
            rgb, nir = rgb[y:y + c, x:x + c], nir[y:y + c, x:x + c]
            if random.random() < 0.5:
                rgb, nir = rgb[:, ::-1].copy(), nir[:, ::-1].copy()
            if random.random() < 0.5:
                rgb, nir = rgb[::-1].copy(), nir[::-1].copy()
            # PHOTOMETRIC augmentation on RGB ONLY (target NIR stays fixed): teaches
            # the model to map many camera renderings -> the same NIR, i.e. to rely
            # on structure/relative cues rather than one camera's absolute RGB. This
            # is our main lever against the A7C II domain gap. Kept mild so the
            # RGB->NIR relationship isn't destroyed.
            if self.photometric:
                gains = np.array([random.uniform(0.85, 1.15) for _ in range(3)], np.float32)
                gamma = random.uniform(0.8, 1.25)
                bias = random.uniform(-0.03, 0.03)
                rgb = np.clip(rgb, 0.0, 1.0) ** gamma
                rgb = np.clip(rgb * gains + bias, 0.0, 1.0)
        else:
            rgb = rgb[:h // 16 * 16, :w // 16 * 16]
            nir = nir[:h // 16 * 16, :w // 16 * 16]
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))
        nir_t = torch.from_numpy(nir[None])
        return rgb_t, nir_t


def find_pairs(data_dir):
    pairs = []
    for rgb_p in sorted(glob.glob(os.path.join(data_dir, "**", "*_rgb.*"), recursive=True)):
        for ext in (".tiff", ".tif", ".png", ".jpg"):
            nir_p = rgb_p.replace("_rgb.", "_nir.").rsplit(".", 1)[0] + ext
            if os.path.exists(nir_p):
                pairs.append((rgb_p, nir_p))
                break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data/nirscene1"],
                    help="one or more dataset roots (combined corpus)")
    ap.add_argument("--manifest", default=None,
                    help="read rgb<TAB>nir pairs from a manifest file (build_manifest.py)")
    ap.add_argument("--no-photometric", action="store_true",
                    help="disable RGB photometric augmentation (camera-robustness)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=25,
                    help="early stop after this many epochs with no val improvement")
    ap.add_argument("--min-delta", type=float, default=5e-4,
                    help="min val-L1 improvement that counts as progress")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "nir_unet.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    pairs = []
    if args.manifest:
        with open(args.manifest) as f:
            pairs = [tuple(line.rstrip("\n").split("\t")) for line in f if "\t" in line]
        print(f"  {len(pairs)} pairs from manifest {args.manifest}")
    else:
        for d in args.data:
            found = find_pairs(d)
            print(f"  {len(found):6d} pairs in {d}")
            pairs += found
    if not pairs:
        sys.exit("no pairs found (check --data / --manifest)")
    random.shuffle(pairs)
    nval = max(1, int(len(pairs) * args.val_frac))
    val_pairs, train_pairs = pairs[:nval], pairs[nval:]
    print(f"{len(pairs)} pairs  ({len(train_pairs)} train / {len(val_pairs)} val)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tl = DataLoader(PairDataset(train_pairs, args.crop, True, not args.no_photometric),
                    batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True)
    vl = DataLoader(PairDataset(val_pairs, args.crop, False), batch_size=1, num_workers=2)

    model = CompactUNet(base=args.base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(dev)
    l1 = nn.L1Loss()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    best = 1e9
    since = 0           # epochs since last meaningful (>min_delta) improvement
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for rgb, nir in tl:
            rgb, nir = rgb.to(dev, non_blocking=True), nir.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                loss = l1(model(rgb), nir)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item()
        model.eval()
        vtot, n = 0.0, 0
        with torch.inference_mode():
            for rgb, nir in vl:
                rgb, nir = rgb.to(dev), nir.to(dev)
                vtot += l1(model(rgb), nir).item(); n += 1
        vmae = vtot / max(1, n)
        since = 0 if vmae < best - args.min_delta else since + 1
        saved = vmae < best
        if saved:
            best = vmae
            torch.save({"model": model.state_dict(), "base": args.base, "val_l1": vmae}, args.out)
        print(f"epoch {ep+1:3d}/{args.epochs}  train L1 {tot/len(tl):.4f}  val L1 {vmae:.4f}"
              + ("  *" if saved else f"  (no improve {since}/{args.patience})"))
        if since >= args.patience:
            print(f"early stop: no val improvement >{args.min_delta} in {args.patience} epochs "
                  f"(stopped at epoch {ep+1}). best val L1 {best:.4f}")
            break
    print(f"done. best val L1 {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
