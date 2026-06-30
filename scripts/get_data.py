#!/usr/bin/env python3
"""
Stage the RGB->NIR training corpus into ~/aurachrome_data/. Download only — never
trains. Designed to stay small (<500 GB) and avoid near-duplicate video frames.

POSTECH is huge (~90k stereo frames); we take only the *undistorted left* RGB+NIR
PNGs and subsample by stride, yielding a few thousand diverse monocular pairs.
Use `--inspect` first (metadata only, no bytes) to verify the file tree + size.

Usage:
    python scripts/get_data.py postech --inspect            # list tree + sizes, no download
    python scripts/get_data.py postech --stride 12 --cap 6000
    python scripts/get_data.py p2irc
    python scripts/get_data.py sugarbeets                   # fetches the batch scripts (guided)
    python scripts/get_data.py hyperskin                    # prints EULA instructions
"""
import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

DATA = Path(os.path.expanduser("~/aurachrome_data"))
POSTECH_REPO = "DivisonOfficer/Pixel-aligned_RGB-NIR_stereo_dataset"


def _human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}PB"


# ---- POSTECH (HuggingFace) -------------------------------------------------
def _postech_files():
    from huggingface_hub import HfApi
    api = HfApi()
    out = []
    for t in api.list_repo_tree(POSTECH_REPO, repo_type="dataset", recursive=True):
        if hasattr(t, "size"):                       # RepoFile (skip RepoFolder)
            out.append((t.path, t.size or 0))
    return out


def _postech_tarballs():
    # Only the sampled_540p tarballs carry the clean <ts>/rgb/left.png + nir/left.png
    # layout (the full-res ones store frames differently and yield nothing). 540p is
    # plenty — we train on 256 crops. Smallest-first so subsets download fast.
    tars = [(p, s) for p, s in _postech_files()
            if p.endswith(".tar.gz") and "sampled_540p" in p]
    return sorted(tars, key=lambda x: x[1])


def postech_inspect():
    tars = _postech_tarballs()
    total = sum(s for _, s in tars)
    print(f"POSTECH repo: {len(tars)} session tarballs, total {_human(total)} "
          f"(avg {_human(total/max(1,len(tars)))}/session)")
    for p, s in tars[:6]:
        print(f"    {p}   {_human(s)}")
    print(f"    ... ({len(tars)} total)")
    print("Plan: download a SUBSET of sessions, extract, keep every Nth frame's "
          "undistorted left RGB+NIR, then delete the tarball — only a few GB stay on disk.")


def postech_download(sessions, frame_stride, cap):
    """Download a strided subset of session tarballs; from each, keep every Nth
    frame's undistorted left RGB+NIR as <frame>_rgb.png/_nir.png (our convention),
    then delete the tarball + extraction to bound disk."""
    import shutil
    import tarfile
    from huggingface_hub import hf_hub_download
    tars = _postech_tarballs()
    step = max(1, len(tars) // sessions)
    chosen = tars[::step][:sessions]
    out = DATA / "postech"
    out.mkdir(parents=True, exist_ok=True)
    print(f"POSTECH: {len(tars)} sessions; taking {len(chosen)} "
          f"(stride {step}), every {frame_stride}th frame, cap {cap} total.")
    kept = 0
    for path, size in chosen:
        if kept >= cap:
            break
        print(f"  downloading {path} ({_human(size)}) ...")
        local = hf_hub_download(POSTECH_REPO, path, repo_type="dataset", local_dir=out / "_tmp")
        ex = out / "_tmp" / "ex"
        ex.mkdir(parents=True, exist_ok=True)
        with tarfile.open(local) as tf:
            tf.extractall(ex)
        # find undistorted left rgb pngs, match to nir
        rgbs = sorted(p for p in ex.rglob("*rgb/left.png"))
        for j, rp in enumerate(rgbs):
            if j % frame_stride:
                continue
            if kept >= cap:
                break
            npth = Path(str(rp).replace("/rgb/", "/nir/"))
            if not npth.exists():
                continue
            stem = f"{Path(path).stem}_{rp.parent.parent.name}"
            shutil.copy(rp, out / f"{stem}_rgb.png")
            shutil.copy(npth, out / f"{stem}_nir.png")
            kept += 1
        shutil.rmtree(out / "_tmp", ignore_errors=True)
        print(f"    kept {kept} pairs so far")
    print(f"POSTECH done: {kept} pairs -> {out}")


# ---- p2irc (GitHub) --------------------------------------------------------
def p2irc_download():
    out = DATA / "p2irc"
    repo = out / "rgb2nir"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/p2irc/rgb2nir", str(repo)], check=True)
    for z in ("trainA.zip", "trainB.zip"):
        zp = repo / z
        if zp.exists():
            dst = out / z[:-4]      # trainA / trainB
            dst.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(dst)
            print(f"  extracted {z} -> {dst}")
    print("p2irc done ->", out)


# ---- Sugar Beets (guided) --------------------------------------------------
def sugarbeets_download():
    print("Sugar Beets 2016 has no clean per-session URL; it's batch-script based.")
    print("Fetching the official download scripts so we can pull ONE small session as the subset:")
    out = DATA / "sugarbeets"
    out.mkdir(parents=True, exist_ok=True)
    url = "https://www.ipb.uni-bonn.de/datasets_IJRR2017/ijrr_download_scripts.zip"
    dst = out / "ijrr_download_scripts.zip"
    subprocess.run(["curl", "-sSL", "-o", str(dst), url], check=True)
    with zipfile.ZipFile(dst) as zf:
        zf.extractall(out)
    print(f"  scripts in {out} — inspect them to pick the smallest recording, then run that one.")


# ---- Hyper-Skin (manual EULA) ---------------------------------------------
def hyperskin_download():
    print("Hyper-Skin is EULA-gated — it CANNOT be auto-downloaded.")
    print("  1) Sign the data agreement at: https://hyper-skin-2023.github.io/")
    print("  2) They email a secure link; download the (RGB, VIS) and (MSI, NIR) sets.")
    print(f"  3) Drop the files into: {DATA/'hyperskin'}")
    print("  Then I'll build the hyperspectral->(RGB, 840nm NIR) extractor for the skin prior.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["postech", "p2irc", "sugarbeets", "hyperskin"])
    ap.add_argument("--inspect", action="store_true", help="POSTECH: list tarballs+sizes, no download")
    ap.add_argument("--sessions", type=int, default=8, help="POSTECH: how many session tarballs to pull")
    ap.add_argument("--frame-stride", type=int, default=15, help="POSTECH: keep every Nth frame/session")
    ap.add_argument("--cap", type=int, default=4000, help="POSTECH: max total pairs")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if args.dataset == "postech":
        postech_inspect() if args.inspect else \
            postech_download(args.sessions, args.frame_stride, args.cap)
    elif args.dataset == "p2irc":
        p2irc_download()
    elif args.dataset == "sugarbeets":
        sugarbeets_download()
    elif args.dataset == "hyperskin":
        hyperskin_download()


if __name__ == "__main__":
    main()
