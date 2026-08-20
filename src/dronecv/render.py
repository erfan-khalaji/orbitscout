"""Fly a new camera through a trained splat.

The point of reconstructing a scene is being able to look at it from somewhere
the drone never was. This loads a trained PLY and renders a synthetic camera
path, which is what makes a splat legible as 3D rather than as a point cloud.

Paths are built in the frame of the original capture: the reconstructed camera
constellation defines a centre, an up axis and a radius, and the synthetic path
is expressed relative to those. That keeps a generated orbit level with the
real flight instead of tumbling through an arbitrary model axis.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch


def load_ply(path: str | Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Read a 3DGS PLY back into raw (unactivated) parameter tensors."""
    path = Path(path)
    with open(path, "rb") as f:
        header, names = [], []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header.append(line)
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * len(names) * 4), dtype=np.float32).reshape(n, len(names))

    col = {nm: i for i, nm in enumerate(names)}
    t = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)

    dc = sorted([k for k in col if k.startswith("f_dc_")], key=lambda s: int(s.split("_")[-1]))
    rest = sorted([k for k in col if k.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))

    sh0 = t(data[:, [col[k] for k in dc]]).reshape(n, 3, 1).transpose(1, 2)
    if rest:
        shN = t(data[:, [col[k] for k in rest]]).reshape(n, 3, -1).transpose(1, 2)
    else:
        shN = torch.zeros((n, 0, 3), device=device)

    return {
        "means": t(data[:, [col["x"], col["y"], col["z"]]]),
        "scales": t(data[:, [col[f"scale_{i}"] for i in range(3)]]),
        "quats": t(data[:, [col[f"rot_{i}"] for i in range(4)]]),
        "opacities": t(data[:, col["opacity"]]),
        "sh0": sh0,
        "shN": shN,
    }


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World-to-camera matrix for a camera at `eye` looking at `target`."""
    f = target - eye
    f = f / max(np.linalg.norm(f), 1e-9)
    r = np.cross(f, up)
    nr = np.linalg.norm(r)
    if nr < 1e-6:  # degenerate when the view direction is parallel to up
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
        nr = np.linalg.norm(r)
    r = r / max(nr, 1e-9)
    u = np.cross(r, f)
    R = np.stack([r, u, f], axis=0)  # rows: camera x, y, z in world
    vm = np.eye(4)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ eye
    return vm


def capture_frame(model_dir: str | Path) -> dict:
    """Centre, up axis and radius of the original camera constellation."""
    from .splat import load_colmap

    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    centers = []
    for img in rec.images.values():
        rigid = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        R = np.asarray(rigid.rotation.matrix())
        t = np.asarray(rigid.translation).reshape(3)
        centers.append(-R.T @ t)
    C = np.stack(centers)
    centroid = C.mean(axis=0)

    # The flight plane is the two dominant directions of the camera spread;
    # its normal is "up" (sign chosen to match the cameras' own up vector).
    d = C - centroid
    _, _, Vt = np.linalg.svd(d, full_matrices=False)
    normal = Vt[2]

    ups = []
    for img in rec.images.values():
        rigid = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        ups.append(-np.asarray(rigid.rotation.matrix())[1])
    mean_up = np.mean(ups, axis=0)
    if np.dot(normal, mean_up) < 0:
        normal = -normal

    radius = float(np.linalg.norm(d - (d @ normal)[:, None] * normal, axis=1).mean())
    K = rec.cameras[next(iter(rec.cameras))].calibration_matrix()
    cam = rec.cameras[next(iter(rec.cameras))]
    return {
        "centroid": centroid, "up": normal, "radius": radius,
        "plane": (Vt[0], Vt[1]), "K": np.asarray(K),
        "width": cam.width, "height": cam.height,
    }


def render_path(
    ply: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    n_frames: int = 240,
    fps: float = 30.0,
    radius_scale: float = 0.85,
    height_scale: float = 0.25,
    turns: float = 1.0,
    downscale: float = 1.0,
    device: str = "cuda",
) -> str:
    """Render a synthetic orbit around the reconstructed scene."""
    import cv2
    from gsplat import rasterization

    g = load_ply(ply, device=device)
    fr = capture_frame(model_dir)

    e0, e1 = fr["plane"]
    up = fr["up"]
    centroid = fr["centroid"]
    R = fr["radius"] * radius_scale

    W = int(fr["width"] * downscale) // 2 * 2
    H = int(fr["height"] * downscale) // 2 * 2
    K = fr["K"].copy()
    K[:2] *= downscale
    Kt = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f"_{out_path.stem}_nv"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    quats = g["quats"] / g["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    colors = torch.cat([g["sh0"], g["shN"]], dim=1)
    sh_deg = int(math.sqrt(colors.shape[1])) - 1

    with torch.inference_mode():
        for i in range(n_frames):
            a = 2 * math.pi * turns * i / n_frames
            # Gentle vertical bob so the move reads as 3D rather than a pan.
            eye = (
                centroid
                + R * (math.cos(a) * e0 + math.sin(a) * e1)
                + up * (fr["radius"] * height_scale * (0.5 + 0.5 * math.sin(a * 2)))
            )
            vm = _look_at(eye, centroid, up)
            vmt = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0)

            img, _, _ = rasterization(
                means=g["means"], quats=quats, scales=torch.exp(g["scales"]),
                opacities=torch.sigmoid(g["opacities"]), colors=colors,
                viewmats=vmt, Ks=Kt, width=W, height=H,
                sh_degree=max(sh_deg, 0),
            )
            arr = (img[0].clamp(0, 1) * 255).byte().cpu().numpy()[:, :, ::-1]
            cv2.imwrite(str(tmp / f"n_{i:05d}.jpg"), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
         "-start_number", "0", "-i", str(tmp / "n_%05d.jpg"),
         "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)],
        check=True,
    )
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return str(out_path)
