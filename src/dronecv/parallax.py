"""Render a 2.5D parallax camera move from frames plus monocular depth.

True 3D wants the camera to have orbited the subject. When it did not -- a
straight dolly, a short clip, a subject made of trees -- a depth-displacement
render still buys convincing volume, because for small virtual baselines the
image-space motion of a point is very nearly proportional to its disparity:

    du  =  f * tx / Z  =  (f * tx) * (1/Z)  =  k * disparity

So a normalised inverse-depth map is directly a displacement field, and a
virtual camera move is a scalar times that field. No calibration is needed;
the amplitude is chosen for looks.

Warping is done forward (source pixel pushed to its destination) with a
z-buffer, because backward warping needs the inverse field, which does not
exist across depth discontinuities. Forward warping instead leaves holes where
background is revealed, and those get filled by a depth-aware pull.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch


def _camera_offset(pattern: str, phase: float) -> tuple[float, float, float]:
    """Virtual camera offset (x, y, zoom) at phase in [0, 1)."""
    a = 2.0 * math.pi * phase
    if pattern == "orbit":
        return math.cos(a), 0.45 * math.sin(a), 0.0
    if pattern == "sway":
        return math.sin(a), 0.0, 0.0
    if pattern == "dolly":
        return 0.0, 0.0, 0.5 * (1.0 - math.cos(a))
    raise ValueError(f"unknown pattern {pattern!r}")


def _forward_warp(
    rgb: torch.Tensor,      # (3,H,W) float 0..1
    disp: torch.Tensor,     # (H,W) float 0..1, 1 = nearest
    dx: float,
    dy: float,
    zoom: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Push every source pixel to its destination, nearest-wins.

    Returns the warped image and a validity mask (1 where a pixel landed).
    """
    _, H, W = rgb.shape
    dev = rgb.device

    yy, xx = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )

    # Displacement is proportional to disparity. Zoom pushes points outward
    # from centre, also scaled by disparity so near things grow faster.
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    u = xx + dx * disp
    v = yy + dy * disp
    if zoom:
        u = u + zoom * disp * (xx - cx)
        v = v + zoom * disp * (yy - cy)

    ui = torch.round(u).long().clamp_(0, W - 1)
    vi = torch.round(v).long().clamp_(0, H - 1)
    flat = (vi * W + ui).reshape(-1)

    # Z-buffer: at each destination keep the largest disparity (nearest).
    zbuf = torch.full((H * W,), -1.0, device=dev)
    zbuf.scatter_reduce_(0, flat, disp.reshape(-1), reduce="amax", include_self=True)

    # A source pixel writes colour only if it owns its destination.
    owns = disp.reshape(-1) >= (zbuf.gather(0, flat) - 1e-6)
    idx = flat[owns]

    out = torch.zeros((3, H * W), device=dev)
    mask = torch.zeros((H * W,), device=dev)
    src = rgb.reshape(3, -1)[:, owns]
    out.scatter_(1, idx.unsqueeze(0).expand(3, -1), src)
    mask.scatter_(0, idx, torch.ones_like(idx, dtype=torch.float32))

    return out.reshape(3, H, W), mask.reshape(H, W)


def _fill_holes(img: torch.Tensor, mask: torch.Tensor, iters: int = 24) -> torch.Tensor:
    """Fill disocclusions by iteratively pulling from valid neighbours.

    Disocclusions sit behind whatever moved, so background colour is the
    correct filler. A blur-and-renormalise pull spreads valid colour inward a
    ring per iteration, which is enough for the few-pixel gaps a small virtual
    baseline opens. On a backlit treeline this reads as a soft edge rather
    than the smeared streaks a naive stretch produces.
    """
    k = 5
    pad = k // 2
    weight = torch.ones((1, 1, k, k), device=img.device) / (k * k)

    filled = img * mask
    m = mask.clone()
    for _ in range(iters):
        if m.min() > 0.5:
            break
        holes = (m < 0.5).float()
        if holes.sum() == 0:
            break
        num = torch.nn.functional.conv2d(
            (filled * m).unsqueeze(0), weight.expand(3, 1, k, k), padding=pad, groups=3
        ).squeeze(0)
        den = torch.nn.functional.conv2d(
            m.unsqueeze(0).unsqueeze(0), weight, padding=pad
        ).squeeze(0).squeeze(0)
        valid = den > 1e-5
        upd = torch.where(valid.unsqueeze(0), num / den.clamp_min(1e-5), filled)
        filled = torch.where((m < 0.5).unsqueeze(0), upd, filled)
        m = torch.where(valid, torch.ones_like(m), m)
    return filled.clamp(0, 1)


def render_parallax(
    frames_dir: str | Path,
    depths_dir: str | Path,
    out_path: str | Path,
    amplitude: float = 0.035,
    n_frames: int = 0,
    fps: float = 30.0,
    pattern: str = "orbit",
    device: str | None = None,
    crop: float = 0.94,
) -> str:
    """Render frames+depth into a parallax move, muxed to H.264 via ffmpeg.

    amplitude is the peak displacement as a fraction of image width.
    crop trims the frame edges, where forward warping has no incoming data.
    """
    frames_dir, depths_dir = Path(frames_dir), Path(depths_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fr = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    dp = sorted(depths_dir.glob("*.npy")) + sorted(depths_dir.glob("*.png"))
    if not fr:
        raise RuntimeError(f"no frames in {frames_dir}")
    if not dp:
        raise RuntimeError(f"no depth maps in {depths_dir}")

    n = min(len(fr), len(dp))
    total = n_frames or n
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tmp = out_path.parent / f"_{out_path.stem}_render"
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    H = W = None
    for i in range(total):
        src = i * n // total
        bgr = cv2.imread(str(fr[src]))
        if bgr is None:
            continue
        rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().to(dev) / 255.0

        d = dp[src]
        raw = np.load(d) if d.suffix == ".npy" else cv2.imread(str(d), cv2.IMREAD_UNCHANGED)
        disp = torch.from_numpy(raw.astype(np.float32)).to(dev)
        if disp.ndim == 3:
            disp = disp[..., 0]
        if disp.shape != rgb.shape[1:]:
            disp = torch.nn.functional.interpolate(
                disp[None, None], size=rgb.shape[1:], mode="bilinear", align_corners=False
            )[0, 0]
        lo, hi = torch.quantile(disp, 0.02), torch.quantile(disp, 0.98)
        disp = ((disp - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)

        if H is None:
            H, W = rgb.shape[1], rgb.shape[2]

        ox, oy, oz = _camera_offset(pattern, i / max(total, 1))
        px = amplitude * W
        try:
            warped, mask = _forward_warp(rgb, disp, ox * px, oy * px, oz * amplitude * 0.6)
            filled = _fill_holes(warped, mask)
        except torch.OutOfMemoryError:
            # An 8 GB card is easily contended. Warping is cheap enough on CPU
            # that degrading beats failing a long render near the end.
            torch.cuda.empty_cache()
            rgb, disp = rgb.cpu(), disp.cpu()
            dev = torch.device("cpu")
            warped, mask = _forward_warp(rgb, disp, ox * px, oy * px, oz * amplitude * 0.6)
            filled = _fill_holes(warped, mask)

        img = (filled.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        if crop < 1.0:
            # round down to even; yuv420p cannot encode odd dimensions
            ch, cw = int(H * crop) & ~1, int(W * crop) & ~1
            y0, x0 = (H - ch) // 2, (W - cw) // 2
            img = img[y0 : y0 + ch, x0 : x0 + cw]
        cv2.imwrite(str(tmp / f"r_{i:05d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
         "-start_number", "0", "-i", str(tmp / "r_%05d.jpg"),
         "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)],
        check=True,
    )
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    return str(out_path)
