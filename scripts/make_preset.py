#!/usr/bin/env python3
"""
Generate Lightroom develop presets (.xmp) that appear in the Presets panel.

Each preset selects the matching Aerochrome camera profile (the .dcp built by
make_profile.py), so one click in the Presets list applies the full look. The
presets are grouped under "Aerochrome".

Install location (Windows):
    %APPDATA%\\Adobe\\CameraRaw\\Settings\\Aerochrome\\
(macOS: ~/Library/Application Support/Adobe/CameraRaw/Settings/Aerochrome/)

Usage:
    python scripts/make_preset.py                 # writes to luts/presets/
    python scripts/make_preset.py --install DIR   # also copies into DIR
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "luts", "presets")

# stable, distinct UUIDs (32 hex chars) per preset.
# grain = (amount, size, frequency/roughness) for Lightroom's native Grain panel.
# Aerochrome is a moderately grainy ISO~200-400 IR reversal stock, so the texture is
# a "little touch", medium-fine -- NOT heavy. Baked into the preset so it rides along
# with one click but stays fully live-editable (Effects -> Grain). LR grain is
# luminance-only; real EIR grain is faintly chromatic (can't be reproduced here).
# Per look: Classic authentic-moderate; Punchy a touch heavier; Muted grainier
# (faded-scan feel); Portrait gentle so skin stays clean.
PRESETS = [
    ("classic",  "Aerochrome Classic v6",  "aec01a55c1a55c1a55c1a55c1a55c106", (25, 22, 50)),
    ("punchy",   "Aerochrome Punchy v6",   "aec0punch1punch1punch1punch10206", (32, 24, 50)),
    ("muted",    "Aerochrome Muted v6",    "aec0mute1mute1mute1mute1mute10306", (38, 25, 45)),
    ("portrait", "Aerochrome Portrait v6", "aec0port1port1port1port1port10406", (16, 20, 50)),
]

TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Aerochrome LUT toolkit">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    crs:Version="15.4"
    crs:ProcessVersion="11.0"
    crs:CameraProfile="{profile}"
    crs:GrainAmount="{grain_amount}"
    crs:GrainSize="{grain_size}"
    crs:GrainFrequency="{grain_freq}"
    crs:HasSettings="True"
    crs:PresetType="Normal"
    crs:Cluster=""
    crs:UUID="{uuid}"
    crs:SupportsAmount="False"
    crs:SupportsColor="True"
    crs:SupportsMonochrome="True"
    crs:SupportsHighDynamicRange="True"
    crs:SupportsNormalDynamicRange="True"
    crs:SupportsSceneReferred="True"
    crs:SupportsOutput="True">
   <crs:Name>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{name}</rdf:li>
    </rdf:Alt>
   </crs:Name>
   <crs:ShortName>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">{name}</rdf:li>
    </rdf:Alt>
   </crs:ShortName>
   <crs:Group>
    <rdf:Alt>
     <rdf:li xml:lang="x-default">Aerochrome</rdf:li>
    </rdf:Alt>
   </crs:Group>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", help="folder to also copy the .xmp presets into")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    written = []
    for _key, name, uuid, grain in PRESETS:
        xmp = TEMPLATE.format(profile=name, name=name, uuid=uuid,
                              grain_amount=grain[0], grain_size=grain[1],
                              grain_freq=grain[2])
        path = os.path.join(OUTDIR, name.replace(" ", "_") + ".xmp")
        with open(path, "w", encoding="utf-8") as f:
            f.write(xmp)
        written.append(path)
        print(f"wrote {path}")

    if args.install:
        dest = os.path.join(args.install, "Aerochrome")
        os.makedirs(dest, exist_ok=True)
        for p in written:
            shutil.copy(p, dest)
        print(f"installed {len(written)} preset(s) -> {dest}")


if __name__ == "__main__":
    main()
