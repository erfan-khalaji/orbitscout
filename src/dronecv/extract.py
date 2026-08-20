"""Extract reconstruction-ready frames from drone video.

Structure-from-motion wants sharp, well-separated views. Naive fixed-interval
sampling grabs whichever frame the interval lands on, which on a moving drone
is frequently motion-blurred; blurred frames poison feature matching and can
break an otherwise recoverable reconstruction.

Strategy: oversample, then keep the sharpest frame in each temporal bucket.
Sharpness is variance of the Laplacian, measured on a downscaled grayscale
copy so the metric tracks real defocus/motion blur rather than sensor noise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ExtractReport:
    video: str
    out_dir: str
    kept: int
    considered: int
    target: int
    mean_sharpness: float
    min_sharpness: float
    rejected_blurry: int
    width: int
    height: int


def probe(video: str | Path) -> dict:
    """Container/stream facts needed to plan extraction."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:stream_side_data=rotation",
        "-show_entries", "format=duration",
        "-of", "json", str(video),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    st = js["stream"][0] if js.get("stream") else js.get("streams", [{}])[0]
    num, _, den = st.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1)
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    return {
        "width": int(st.get("width", 0)),
        "height": int(st.get("height", 0)),
        "fps": fps,
        "duration": float(js.get("format", {}).get("duration", 0.0)),
        "rotation": rot,
    }


def _sharpness(bgr: np.ndarray) -> float:
    """Variance of the Laplacian on a downscaled gray copy."""
    h, w = bgr.shape[:2]
    scale = 512.0 / max(h, w)
    small = cv2.resize(bgr, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(
    video: str | Path,
    out_dir: str | Path,
    target: int = 150,
    oversample: int = 3,
    long_edge: int = 1600,
    quality: int = 2,
    start: float = 0.0,
    end: float | None = None,
) -> ExtractReport:
    """Decode `target * oversample` candidates, keep the sharpest per bucket.

    long_edge caps the written resolution. 8 GB of VRAM will not train a splat
    on native 4 K, and SfM converges fine at ~1600 px, so downscaling here saves
    hours downstream rather than costing accuracy.
    """
    video = Path(video)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe(video)
    dur = info["duration"]
    end = min(end, dur) if end else dur
    span = max(end - start, 1e-3)

    n_cand = max(target * oversample, target)
    # Sample candidate timestamps uniformly across the usable span.
    stamps = [start + span * (i + 0.5) / n_cand for i in range(n_cand)]

    # Single sequential decode is dramatically faster than N random seeks:
    # 4K HEVC seeking costs a keyframe hunt plus full re-decode every time, so
    # hundreds of seeks take minutes where one linear pass takes seconds.
    # ffmpeg emits candidates at a fixed rate into a temp dir, and sharpness
    # selection then runs over those decoded JPEGs.
    cand_fps = n_cand / span
    tmp = out_dir / "_candidates"
    tmp.mkdir(parents=True, exist_ok=True)

    vf = [f"fps={cand_fps:.6f}"]
    if long_edge:
        vf.append(
            f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
        )

    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", str(video)]
    if end and end < dur:
        cmd += ["-t", f"{max(end - start, 0.001)}"]
    cmd += ["-vf", ",".join(vf), "-q:v", "2", str(tmp / "cand_%05d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True)

    cand_files = sorted(tmp.glob("cand_*.jpg"))
    considered = len(cand_files)
    if not cand_files:
        raise RuntimeError(f"ffmpeg produced no frames from {video}")

    # Bucket candidates into `target` groups; keep the sharpest of each.
    per_bucket = max(1, considered // max(target, 1))
    best: list[tuple[float, int, Path]] = []
    sharps: list[float] = []
    cur_best: tuple[float, Path] | None = None
    kept = 0

    for idx, cf in enumerate(cand_files):
        frame = cv2.imread(str(cf))
        if frame is None:
            continue
        s_val = _sharpness(frame)
        if cur_best is None or s_val > cur_best[0]:
            cur_best = (s_val, cf)
        if (idx + 1) % per_bucket == 0 and cur_best is not None:
            best.append((cur_best[0], kept, cur_best[1]))
            sharps.append(cur_best[0])
            kept += 1
            cur_best = None

    if cur_best is not None:
        best.append((cur_best[0], kept, cur_best[1]))
        sharps.append(cur_best[0])
        kept += 1

    # Drop frames far below the median: these are the unrecoverable ones.
    floor = float(np.median(sharps)) * 0.35 if sharps else 0.0
    rejected = 0
    written = 0
    rot = info["rotation"]

    frame = None
    for s, _, cf in best:
        if s < floor:
            rejected += 1
            continue
        frame = cv2.imread(str(cf))
        if frame is None:
            rejected += 1
            continue
        # No manual rotation here: ffmpeg applies the container display matrix
        # during decode by default, so candidates are already upright. Rotating
        # again would turn DJI portrait clips back to landscape.

        h, w = frame.shape[:2]
        if long_edge and max(h, w) > long_edge:
            sc = long_edge / max(h, w)
            frame = cv2.resize(frame, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)

        cv2.imwrite(
            str(out_dir / f"frame_{written:05d}.jpg"),
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 100 - quality * 3],
        )
        written += 1

    shutil.rmtree(tmp, ignore_errors=True)
    final = frame.shape[:2] if frame is not None else (0, 0)
    return ExtractReport(
        video=str(video),
        out_dir=str(out_dir),
        kept=written,
        considered=considered,
        target=target,
        mean_sharpness=round(float(np.mean(sharps)) if sharps else 0.0, 1),
        min_sharpness=round(float(np.min(sharps)) if sharps else 0.0, 1),
        rejected_blurry=rejected,
        width=final[1],
        height=final[0],
    )
