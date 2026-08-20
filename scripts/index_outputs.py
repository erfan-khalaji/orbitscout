#!/usr/bin/env python3
"""Summarise everything the pipeline produced, so results are easy to find.

Outputs deliberately live outside the repo (they are large and derived).
This walks an output root and prints what is there, what made it, and how
to look at it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

KIND = {
    "manifest.json": "frame -> source-timestamp map (feeds GPS cross-check)",
    "sfm_report.json": "structure-from-motion result",
    "depth_report.json": "monocular depth run",
    "splat_report.json": "gaussian splatting training",
    "geo_report.json": "SfM vs GPS alignment",
    "points.ply": "SfM sparse point cloud",
    "splat.ply": "trained gaussian splat (open in any splat viewer)",
}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def vid_info(p: Path) -> str:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(p)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        vals = ",".join(out).strip(",").split(",")
        if len(vals) >= 3:
            return f"{vals[0]}x{vals[1]}, {float(vals[2]):.1f}s"
    except Exception:
        pass
    return ""


def main(root: str) -> int:
    root = Path(root)
    if not root.exists():
        print(f"no such output root: {root}", file=sys.stderr)
        return 1

    print(f"OUTPUT ROOT  {root}")
    print(f"total size   {human(sum(f.stat().st_size for f in root.rglob('*') if f.is_file()))}\n")

    for scene in sorted(p for p in root.iterdir() if p.is_dir()):
        files = list(scene.rglob("*"))
        n_frames = sum(1 for f in files if f.suffix.lower() in {".jpg", ".png"})
        print(f"[{scene.name}]  {n_frames} images, {human(sum(f.stat().st_size for f in files if f.is_file()))}")

        for rep in sorted(scene.rglob("*_report.json")) + sorted(scene.glob("manifest.json")):
            try:
                d = json.loads(rep.read_text())
            except Exception:
                continue
            label = KIND.get(rep.name, rep.name)
            print(f"   {rep.name:20} {label}")
            keys = ("registered", "registration_rate", "points3D", "mean_reproj_error_px",
                    "final_gaussians", "final_psnr", "peak_vram_gb", "frames",
                    "temporal_flicker", "model", "metric_scale", "n_matched")
            bits = [f"{k}={d[k]}" for k in keys if k in d]
            if bits:
                print(f"      {', '.join(bits)}")

        for v in sorted(scene.glob("*.mp4")):
            print(f"   {v.name:20} video  {vid_info(v)}  {human(v.stat().st_size)}")
        for ply in sorted(scene.rglob("*.ply")):
            print(f"   {ply.name:20} {KIND.get(ply.name,'point data')}  {human(ply.stat().st_size)}")
        print()

    print("view a .ply splat at https://superspl.at/editor (drag and drop, runs locally)")
    print("view a .ply point cloud with MeshLab, CloudCompare, or Blender import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
