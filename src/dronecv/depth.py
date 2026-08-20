"""Monocular depth over a video clip, with temporal stabilisation.

Depth Anything V2 predicts *relative inverse* depth per image: larger values
are nearer, and the scale is arbitrary and independent per frame. Run naively
over video that independence shows up as flicker -- the treeline pulses toward
and away from camera as the per-frame normalisation drifts.

Two cheap fixes, both applied here:

1. Global normalisation. Percentiles are pooled across a sample of the whole
   clip and one fixed (lo, hi) is applied to every frame, so a frame is never
   rescaled just because its own content changed.
2. Temporal EMA. A causal exponential filter is run forward then backward,
   which removes flicker without the half-frame lag a purely causal filter
   would introduce.

Weights are free to download. The Small checkpoint is Apache-2.0; Base and
Large are CC-BY-NC-4.0 -- free for research/personal use, not for commercial
work. Small is the default here for that reason.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

MODELS = {
    "vits": ("depth-anything/Depth-Anything-V2-Small-hf", "Apache-2.0"),
    "vitb": ("depth-anything/Depth-Anything-V2-Base-hf", "CC-BY-NC-4.0"),
    "vitl": ("depth-anything/Depth-Anything-V2-Large-hf", "CC-BY-NC-4.0"),
}


def _decode_frames(video: Path, out_dir: Path, long_edge: int, max_frames: int) -> list[Path]:
    """Decode the clip to JPEGs once, upright, at the working resolution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    vf = [
        f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
    ]
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
           "-vf", ",".join(vf), "-q:v", "2"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [str(out_dir / "f_%05d.jpg")]
    subprocess.run(cmd, check=True, capture_output=True)
    return sorted(out_dir.glob("f_*.jpg"))


def _ema_bidirectional(stack: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Forward-backward exponential smoothing along axis 0 (zero phase lag)."""
    if stack.shape[0] < 3:
        return stack
    fwd = np.empty_like(stack)
    acc = stack[0].copy()
    for i in range(stack.shape[0]):
        acc = alpha * stack[i] + (1 - alpha) * acc
        fwd[i] = acc
    bwd = np.empty_like(stack)
    acc = fwd[-1].copy()
    for i in range(stack.shape[0] - 1, -1, -1):
        acc = alpha * fwd[i] + (1 - alpha) * acc
        bwd[i] = acc
    return bwd


def run_depth(
    video: str | Path,
    out_dir: str | Path,
    encoder: str = "vits",
    max_frames: int = 0,
    long_edge: int = 1280,
    batch: int = 4,
    smooth: float = 0.6,
) -> dict:
    """Predict stabilised inverse-depth for every frame; write .npy + preview."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    video = Path(video)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    depth_dir = out_dir / "depth"
    prev_dir = out_dir / "depth_preview"
    for d in (depth_dir, prev_dir):
        d.mkdir(parents=True, exist_ok=True)

    frames = _decode_frames(video, frames_dir, long_edge, max_frames)
    if not frames:
        raise RuntimeError(f"no frames decoded from {video}")

    model_id, license_ = MODELS[encoder]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_id, torch_dtype=torch.float16 if dev == "cuda" else torch.float32
    ).to(dev).eval()

    raw: list[np.ndarray] = []
    with torch.inference_mode():
        for i in range(0, len(frames), batch):
            chunk = frames[i : i + batch]
            imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in chunk]
            inputs = proc(images=imgs, return_tensors="pt").to(dev)
            if dev == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            pred = model(**inputs).predicted_depth  # (B,h,w), larger = nearer
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1).float(),
                size=imgs[0].shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            raw.extend(p.cpu().numpy().astype(np.float32) for p in pred)

    stack = np.stack(raw, axis=0)

    # Global normalisation: pool percentiles over a subsample of the clip so
    # every frame shares one scale.
    sample = stack[:: max(1, len(stack) // 40)]
    lo = float(np.percentile(sample, 2.0))
    hi = float(np.percentile(sample, 98.0))
    stack = np.clip((stack - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    if smooth > 0:
        stack = _ema_bidirectional(stack, alpha=smooth)

    for i, d in enumerate(stack):
        np.save(depth_dir / f"d_{i:05d}.npy", d.astype(np.float32))
        cv2.imwrite(
            str(prev_dir / f"d_{i:05d}.jpg"),
            cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )

    # Flicker metric: mean absolute frame-to-frame change, lower is steadier.
    flicker = float(np.mean(np.abs(np.diff(stack, axis=0)))) if len(stack) > 1 else 0.0

    report = {
        "video": str(video),
        "model": model_id,
        "license": license_,
        "frames": len(frames),
        "resolution": [int(stack.shape[2]), int(stack.shape[1])],
        "global_lo": round(lo, 4),
        "global_hi": round(hi, 4),
        "temporal_flicker": round(flicker, 5),
        "frames_dir": str(frames_dir),
        "depth_dir": str(depth_dir),
        "preview_dir": str(prev_dir),
    }
    (out_dir / "depth_report.json").write_text(json.dumps(report, indent=2))
    return report
