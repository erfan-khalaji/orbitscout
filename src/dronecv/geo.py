"""Cross-check an SfM reconstruction against GPS telemetry.

Structure-from-motion recovers geometry only up to a similarity: the model is
correct in shape but arbitrary in scale, orientation and origin. The drone
already logged where it actually was, so fitting the similarity that best maps
recovered camera centres onto the GPS track does two useful things at once.

It recovers **metric scale** -- the factor converting model units to metres,
which is what makes measurements off a splat meaningful -- and it provides an
**independent accuracy check**, because GPS was never an input to the solve.
A low residual means two independent sensors agree on the flight path.

Uses Umeyama (1991) for the closed-form least-squares similarity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping src onto dst. Returns (scale, R, t)."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (sc**2).sum() / n
    scale = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def camera_centers(model_dir: str | Path) -> dict[str, np.ndarray]:
    """World-space camera centre per image name: C = -R^T t."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    out: dict[str, np.ndarray] = {}
    for img in rec.images.values():
        rigid = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        R = np.asarray(rigid.rotation.matrix())
        t = np.asarray(rigid.translation).reshape(3)
        out[img.name] = -R.T @ t
    return out


def compare_to_gps(
    model_dir: str | Path, manifest_path: str | Path, srt_path: str | Path
) -> dict:
    """Align recovered camera track to the GPS track; report scale and residual."""
    from .telemetry import parse_srt

    manifest = json.loads(Path(manifest_path).read_text())
    tel = parse_srt(srt_path)
    fixed = tel.fixed
    if not fixed:
        return {"ok": False, "reason": "no GPS fixes in telemetry"}

    enu = np.array(tel.enu(stride=1), dtype=np.float64)
    times = np.array([f.t for f in fixed], dtype=np.float64)

    centers = camera_centers(model_dir)
    by_name = {e["frame"]: e["t"] for e in manifest["frames"]}

    src, dst, used = [], [], []
    for name, C in centers.items():
        t = by_name.get(name)
        if t is None:
            continue
        # Nearest telemetry sample to this frame's source timestamp.
        i = int(np.argmin(np.abs(times - t)))
        if abs(times[i] - t) > 0.5:
            continue
        src.append(C)
        dst.append(enu[i])
        used.append(name)

    if len(src) < 8:
        return {"ok": False, "reason": f"only {len(src)} frames matched to GPS"}

    src = np.array(src)
    dst = np.array(dst)
    scale, R, tr = umeyama(src, dst)
    pred = (scale * (R @ src.T)).T + tr
    err = np.linalg.norm(pred - dst, axis=1)

    return {
        "ok": True,
        "n_matched": len(src),
        "metric_scale": round(scale, 6),
        "residual_m": {
            "mean": round(float(err.mean()), 3),
            "median": round(float(np.median(err)), 3),
            "p90": round(float(np.percentile(err, 90)), 3),
            "max": round(float(err.max()), 3),
        },
        "gps_track_extent_m": round(float(np.linalg.norm(dst - dst.mean(axis=0), axis=1).max()), 2),
        "model_extent_units": round(float(np.linalg.norm(src - src.mean(axis=0), axis=1).max()), 4),
    }
