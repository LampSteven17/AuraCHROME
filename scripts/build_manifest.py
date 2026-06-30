#!/usr/bin/env python3
"""
Scan ~/aurachrome_data and emit a unified RGB->NIR pair manifest (one
"rgb_path<TAB>nir_path" per line) across all downloaded sources, each with its
own pairing rule. The trainer reads this with `--manifest`.

    python scripts/build_manifest.py            # -> ~/aurachrome_data/manifest.txt
"""
import glob
import os
import sys

DATA = os.path.expanduser("~/aurachrome_data")


def pairs_rgb_nir(root):
    """EPFL / POSTECH convention: <stem>_rgb.<ext> next to <stem>_nir.<ext>."""
    out = []
    for rgb in glob.glob(os.path.join(root, "**", "*_rgb.*"), recursive=True):
        stem = rgb.rsplit("_rgb.", 1)[0]
        for ext in (".png", ".tiff", ".tif", ".jpg"):
            nir = stem + "_nir" + ext
            if os.path.exists(nir):
                out.append((rgb, nir))
                break
    return out


def pairs_trainA_trainB(root):
    """p2irc convention: trainA/<name> (RGB) paired to trainB/<name> (NIR).
    Handles the nested trainA/trainA + trainB/trainB layout from the zips."""
    out = []
    seen = set()
    for a_dir in glob.glob(os.path.join(root, "**", "trainA"), recursive=True):
        b_dir = a_dir.replace("trainA", "trainB")     # mirror every level
        if not os.path.isdir(b_dir):
            continue
        for rgb in sorted(glob.glob(os.path.join(a_dir, "*.png"))
                          + glob.glob(os.path.join(a_dir, "*.jpg"))):
            nir = os.path.join(b_dir, os.path.basename(rgb))
            if os.path.exists(nir) and rgb not in seen:
                out.append((rgb, nir))
                seen.add(rgb)
    return out


SOURCES = {
    "epfl":       pairs_rgb_nir,
    "postech":    pairs_rgb_nir,
    "sugarbeets": pairs_rgb_nir,   # adapter finalized once a session is extracted
    "p2irc":      pairs_trainA_trainB,
}


def main():
    out_path = os.path.join(DATA, "manifest.txt")
    total = 0
    with open(out_path, "w") as f:
        for name, fn in SOURCES.items():
            root = os.path.join(DATA, name)
            if not os.path.isdir(root):
                continue
            pairs = fn(root)
            for rgb, nir in pairs:
                f.write(f"{rgb}\t{nir}\n")
            print(f"  {name:11s} {len(pairs):7d} pairs")
            total += len(pairs)
    print(f"manifest: {total} pairs -> {out_path}")


if __name__ == "__main__":
    main()
