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
    """World-to-camera matrix in COLMAP/gsplat convention.

    That convention is x right, **y down**, z forward. The rows of the rotation
    are those three camera axes expressed in world coordinates. Getting the y
    sign wrong renders the whole scene upside down, which is easy to miss
    because a symmetric orbit still "looks like something".
    """
    f = target - eye
    f = f / max(np.linalg.norm(f), 1e-9)

    r = np.cross(f, up)
    nr = np.linalg.norm(r)
    if nr < 1e-6:  # view direction parallel to up
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
        nr = np.linalg.norm(r)
    r = r / max(nr, 1e-9)

    # x cross z = -y, so cross(r, f) points *up*; camera y is down, hence the
    # negation.
    d = -np.cross(r, f)
    d = d / max(np.linalg.norm(d), 1e-9)

    R = np.stack([r, d, f], axis=0)
    vm = np.eye(4)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ eye
    return vm


def capture_frame(model_dir: str | Path) -> dict:
    """Centre, up axis and orbit radius of the original capture.

    Up is taken as the mean of the registered cameras' own up vectors rather
    than the normal of the camera-constellation plane. The plane normal is
    ambiguous in sign and degenerates badly when a flight is not perfectly
    planar; the cameras were all roughly level, so their mean up is both
    better conditioned and unambiguous.

    The look-at target is the centroid of the *points*, not the cameras --
    that is what the flight was actually pointed at.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))

    centers, ups, fwds = [], [], []
    for img in rec.images.values():
        rigid = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        R = np.asarray(rigid.rotation.matrix())
        t = np.asarray(rigid.translation).reshape(3)
        centers.append(-R.T @ t)
        ups.append(-R[1])   # camera y is down, so world up is -row1
        fwds.append(R[2])
    C = np.stack(centers)
    cam_centroid = C.mean(axis=0)

    up = np.mean(np.stack(ups), axis=0)
    up = up / max(np.linalg.norm(up), 1e-9)

    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float64)
    # Trim outliers before taking the target; SfM always leaves a few strays
    # far from the scene that would drag the centroid off the subject.
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    core = xyz[np.all((xyz >= lo) & (xyz <= hi), axis=1)]
    target = core.mean(axis=0) if len(core) > 50 else xyz.mean(axis=0)

    # Orbit basis: two directions spanning the plane perpendicular to up.
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, up)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e0 = np.cross(up, a)
    e0 /= max(np.linalg.norm(e0), 1e-9)
    e1 = np.cross(up, e0)
    e1 /= max(np.linalg.norm(e1), 1e-9)

    d = C - target
    radial = d - np.outer(d @ up, up)
    radius = float(np.linalg.norm(radial, axis=1).mean())
    height = float(np.mean(d @ up))

    cam = rec.cameras[next(iter(rec.cameras))]
    return {
        "centroid": target,
        "cam_centroid": cam_centroid,
        "up": up,
        "radius": radius,
        "height": height,
        "plane": (e0, e1),
        "K": np.asarray(cam.calibration_matrix()),
        "width": cam.width,
        "height_px": cam.height,
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
    H = int(fr["height_px"] * downscale) // 2 * 2
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
            # Stay at the height the drone actually flew, with a gentle bob,
            # so the synthetic move stays inside the region the splat was
            # supervised on. Straying far from the training views is what
            # produces smeared gaussians.
            bob = fr["radius"] * height_scale * 0.15 * math.sin(a * 2)
            eye = (
                centroid
                + R * (math.cos(a) * e0 + math.sin(a) * e1)
                + up * (fr["height"] + bob)
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


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Shortest-arc quaternion interpolation (wxyz)."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-9)
    th0 = math.acos(max(min(d, 1.0), -1.0))
    th = th0 * u
    q2 = q1 - q0 * d
    q2 /= max(np.linalg.norm(q2), 1e-9)
    return q0 * math.cos(th) + q2 * math.sin(th)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def training_poses(model_dir: str | Path):
    """Registered camera poses in capture order, as (centre, quaternion)."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    rows = []
    for img in rec.images.values():
        rg = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        R = np.asarray(rg.rotation.matrix())
        t = np.asarray(rg.translation).reshape(3)
        rows.append((img.name, -R.T @ t, _R_to_quat(R)))
    rows.sort(key=lambda r: r[0])
    return rows


def render_follow(
    ply: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    n_frames: int = 240,
    fps: float = 30.0,
    offset: float = 0.06,
    smooth: int = 9,
    downscale: float = 1.0,
    device: str = "cuda",
) -> str:
    """Fly a smoothed, laterally-offset version of the original trajectory.

    A full synthetic orbit only makes sense when the capture actually
    surrounded the subject. It frequently did not: a drone "orbit" of a
    distant subject covers very little angle around the scene centroid, and a
    camera swung around that centroid lands in viewpoints the splat was never
    supervised on, which renders as smeared gaussians.

    Following the real path instead keeps every frame inside the supervised
    region, while the smoothing and the lateral offset ensure no rendered view
    is one the drone actually took -- which is what makes it novel-view
    synthesis rather than playback.
    """
    import cv2
    from gsplat import rasterization

    g = load_ply(ply, device=device)
    poses = training_poses(model_dir)
    if len(poses) < 4:
        raise RuntimeError(f"need at least 4 registered views, have {len(poses)}")

    fr = capture_frame(model_dir)
    C = np.stack([p[1] for p in poses])
    Q = np.stack([p[2] for p in poses])

    # Moving-average the positions: raw SfM centres jitter frame to frame, and
    # that jitter is very visible in a synthetic fly-through.
    if smooth > 1:
        k = min(smooth, len(C) // 2 * 2 + 1)
        pad = k // 2
        padded = np.pad(C, ((pad, pad), (0, 0)), mode="edge")
        ker = np.ones(k) / k
        C = np.stack([np.convolve(padded[:, i], ker, mode="valid") for i in range(3)], axis=1)

    up = fr["up"]
    scale = float(np.linalg.norm(C - C.mean(axis=0), axis=1).max())

    W = int(fr["width"] * downscale) // 2 * 2
    H = int(fr["height_px"] * downscale) // 2 * 2
    K = fr["K"].copy()
    K[:2] *= downscale
    Kt = torch.tensor(K, dtype=torch.float32, device=device).unsqueeze(0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f"_{out_path.stem}_fw"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    quats = g["quats"] / g["quats"].norm(dim=-1, keepdim=True).clamp_min(1e-8)
    colors = torch.cat([g["sh0"], g["shN"]], dim=1)
    sh_deg = int(math.sqrt(colors.shape[1])) - 1

    with torch.inference_mode():
        for i in range(n_frames):
            u = i / max(n_frames - 1, 1) * (len(C) - 1)
            j = min(int(u), len(C) - 2)
            fr_u = u - j

            pos = C[j] * (1 - fr_u) + C[j + 1] * fr_u
            q = _slerp(Q[j], Q[j + 1], fr_u)
            R = _quat_to_R(q)

            # Offset sideways and vertically, in the camera's own frame, by a
            # fraction of the trajectory extent.
            right = R[0]
            a = 2 * math.pi * i / max(n_frames, 1)
            pos = pos + right * (offset * scale * math.sin(a)) \
                      + up * (offset * scale * 0.4 * math.sin(2 * a))

            vm = np.eye(4)
            vm[:3, :3] = R
            vm[:3, 3] = -R @ pos
            vmt = torch.tensor(vm, dtype=torch.float32, device=device).unsqueeze(0)

            img, _, _ = rasterization(
                means=g["means"], quats=quats, scales=torch.exp(g["scales"]),
                opacities=torch.sigmoid(g["opacities"]), colors=colors,
                viewmats=vmt, Ks=Kt, width=W, height=H,
                sh_degree=max(sh_deg, 0),
            )
            arr = (img[0].clamp(0, 1) * 255).byte().cpu().numpy()[:, :, ::-1]
            cv2.imwrite(str(tmp / f"f_{i:05d}.jpg"), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
         "-start_number", "0", "-i", str(tmp / "f_%05d.jpg"),
         "-c:v", "libx264", "-preset", "slow", "-crf", "18",
         "-pix_fmt", "yuv420p", str(out_path)],
        check=True,
    )
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return str(out_path)
