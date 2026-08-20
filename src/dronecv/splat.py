"""Compact 3D Gaussian Splatting trainer on top of gsplat's rasteriser.

Deliberately self-contained: gsplat ships the differentiable rasteriser as a
library but keeps its training loop in an examples directory that is not part
of the wheel, so depending on it means vendoring a moving target. The loop here
is small enough to read in one sitting and covers what actually matters --
initialisation from SfM points, an L1+SSIM objective, and adaptive density
control.

Memory is the binding constraint on an 8 GB laptop card. Two levers matter:
`max_gaussians` hard-caps the primitive count during densification, and images
are held on CPU and moved per-iteration rather than resident on device.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SplatConfig:
    iters: int = 7000
    sh_degree: int = 3
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4
    ssim_weight: float = 0.2
    warmup: int = 500
    densify_until: int = 4500
    densify_every: int = 100
    opacity_reset_every: int = 3000
    grad_threshold: float = 2e-4      # absolute floor, secondary guard
    grad_percentile: float = 96.0     # densify the top (100-p)% by gradient
    min_opacity: float = 5e-3
    max_gaussians: int = 900_000
    scene_scale: float = 1.0
    device: str = "cuda"
    eval_every: int = 0
    seed: int = 0


def _quat_norm(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _ssim(a: torch.Tensor, b: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Mean SSIM over a CHW pair, gaussian-windowed."""
    c1, c2 = 0.01**2, 0.03**2
    g = torch.arange(window, dtype=torch.float32, device=a.device) - window // 2
    g = torch.exp(-(g**2) / (2 * 1.5**2))
    g = (g / g.sum()).unsqueeze(0)
    k = (g.T @ g).expand(a.shape[0], 1, window, window).contiguous()
    pad = window // 2
    mu_a = F.conv2d(a.unsqueeze(0), k, padding=pad, groups=a.shape[0])
    mu_b = F.conv2d(b.unsqueeze(0), k, padding=pad, groups=a.shape[0])
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d((a * a).unsqueeze(0), k, padding=pad, groups=a.shape[0]) - mu_a2
    sb = F.conv2d((b * b).unsqueeze(0), k, padding=pad, groups=a.shape[0]) - mu_b2
    sab = F.conv2d((a * b).unsqueeze(0), k, padding=pad, groups=a.shape[0]) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sa + sb + c2)
    return (num / den).mean()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """Inverse of the DC spherical-harmonic evaluation used by 3DGS."""
    return (rgb - 0.5) / 0.28209479177387814


def sh0_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return (sh * 0.28209479177387814 + 0.5).clamp(0, 1)


@dataclass
class GaussianModel:
    means: torch.Tensor
    scales: torch.Tensor      # log-space
    quats: torch.Tensor
    opacities: torch.Tensor   # logit-space
    sh0: torch.Tensor         # (N,1,3)
    shN: torch.Tensor         # (N,K,3)
    params: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.means.shape[0]

    def tensors(self) -> list[torch.Tensor]:
        return [self.means, self.scales, self.quats, self.opacities, self.sh0, self.shN]


def init_from_points(
    xyz: np.ndarray, rgb: np.ndarray, cfg: SplatConfig
) -> GaussianModel:
    """Seed one gaussian per SfM point, sized by local point spacing."""
    dev = cfg.device
    means = torch.tensor(xyz, dtype=torch.float32, device=dev)
    colors = torch.tensor(rgb, dtype=torch.float32, device=dev).clamp(0, 1)
    n = means.shape[0]

    # Initial radius = mean distance to the 3 nearest neighbours, so density
    # follows the reconstruction rather than a global constant.
    with torch.no_grad():
        sample = means[torch.randperm(n, device=dev)[: min(n, 6000)]]
        d = torch.cdist(means, sample)
        knn = d.topk(k=min(4, sample.shape[0]), largest=False).values[:, 1:]
        spacing = knn.mean(dim=1).clamp_min(1e-6)
    scales = torch.log(spacing).unsqueeze(-1).repeat(1, 3)

    quats = torch.zeros((n, 4), device=dev)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), 0.1, device=dev))

    n_sh = (cfg.sh_degree + 1) ** 2
    sh0 = rgb_to_sh0(colors).unsqueeze(1)
    shN = torch.zeros((n, n_sh - 1, 3), device=dev)

    m = GaussianModel(
        means=means.requires_grad_(True),
        scales=scales.requires_grad_(True),
        quats=quats.requires_grad_(True),
        opacities=opacities.requires_grad_(True),
        sh0=sh0.requires_grad_(True),
        shN=shN.requires_grad_(True),
    )
    return m


def make_optimizer(m: GaussianModel, cfg: SplatConfig, scene_scale: float):
    return torch.optim.Adam(
        [
            {"params": [m.means], "lr": cfg.lr_means * scene_scale, "name": "means"},
            {"params": [m.scales], "lr": cfg.lr_scales, "name": "scales"},
            {"params": [m.quats], "lr": cfg.lr_quats, "name": "quats"},
            {"params": [m.opacities], "lr": cfg.lr_opacities, "name": "opacities"},
            {"params": [m.sh0], "lr": cfg.lr_sh0, "name": "sh0"},
            {"params": [m.shN], "lr": cfg.lr_shN, "name": "shN"},
        ],
        eps=1e-15,
    )


def export_ply(m: GaussianModel, path: str | Path) -> str:
    """Write the standard 3DGS PLY consumed by every splat viewer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        xyz = m.means.detach().cpu().numpy()
        f_dc = m.sh0.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        f_rest = m.shN.detach().transpose(1, 2).flatten(start_dim=1).cpu().numpy()
        opac = m.opacities.detach().unsqueeze(-1).cpu().numpy()
        scale = m.scales.detach().cpu().numpy()
        rot = _quat_norm(m.quats.detach()).cpu().numpy()

    n = xyz.shape[0]
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2"]
    names += ["rot_0", "rot_1", "rot_2", "rot_3"]

    data = np.concatenate(
        [xyz, np.zeros_like(xyz), f_dc, f_rest, opac, scale, rot], axis=1
    ).astype(np.float32)

    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {n}\n".encode())
        for nm in names:
            f.write(f"property float {nm}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())
    return str(path)


# --------------------------------------------------------------------------
# adaptive density control
# --------------------------------------------------------------------------


def _reindex_adam(opt, m: GaussianModel, new_tensors: list[torch.Tensor], keep=None, n_new=0):
    """Swap the model tensors inside the optimiser, carrying Adam state across.

    Adam keeps per-element moment estimates. When gaussians are pruned the
    moments must be gathered by the same mask; when gaussians are added the
    moments must be extended with zeros. Rebuilding the optimiser from scratch
    instead would discard momentum and visibly stall training after every
    densification step.
    """
    for group, new in zip(opt.param_groups, new_tensors):
        old = group["params"][0]
        st = opt.state.get(old, None)
        if st is not None:
            ea, eas = st["exp_avg"], st["exp_avg_sq"]
            if keep is not None:
                ea, eas = ea[keep], eas[keep]
            if n_new:
                pad = list(ea.shape)
                pad[0] = n_new
                ea = torch.cat([ea, torch.zeros(pad, device=ea.device)], dim=0)
                eas = torch.cat([eas, torch.zeros(pad, device=eas.device)], dim=0)
            del opt.state[old]
            opt.state[new] = {"step": st["step"], "exp_avg": ea, "exp_avg_sq": eas}
        group["params"] = [new]

    m.means, m.scales, m.quats, m.opacities, m.sh0, m.shN = new_tensors


def _prune(opt, m: GaussianModel, keep: torch.Tensor):
    new = [t.detach()[keep].requires_grad_(True) for t in m.tensors()]
    _reindex_adam(opt, m, new, keep=keep)


def _append(opt, m: GaussianModel, extra: list[torch.Tensor]):
    new = [
        torch.cat([t.detach(), e], dim=0).requires_grad_(True)
        for t, e in zip(m.tensors(), extra)
    ]
    _reindex_adam(opt, m, new, n_new=extra[0].shape[0])


@torch.no_grad()
def densify_and_prune(
    opt, m: GaussianModel, grads: torch.Tensor, cfg: SplatConfig, extent: float
) -> dict:
    """Clone under-sized, split over-sized, then drop transparent gaussians.

    3DGS's core insight: a large view-space positional gradient means the region
    is under-reconstructed. Small gaussians there are *cloned* (the region needs
    more coverage); large ones are *split* (the region needs more detail).
    """
    dev = m.means.device
    n0 = m.n
    big_thresh = 0.01 * extent
    scales = torch.exp(m.scales)

    # Select by percentile rather than a fixed magnitude. The absolute
    # threshold from the 3DGS paper assumes that implementation's gradient
    # convention and normalisation; gsplat's absgrad differs in scale, and a
    # mis-set constant silently disables densification entirely (the failure
    # looks like "training works but the model never gets denser"). A
    # percentile adapts to whatever scale the rasteriser actually produces.
    seen = grads[grads > 0]
    if seen.numel() < 32:
        return {"cloned": 0, "split": 0, "pruned": 0, "n_before": n0, "n_after": n0}
    cutoff = torch.quantile(seen.float(), cfg.grad_percentile / 100.0)
    selected = grads >= torch.clamp(cutoff, min=1e-12)
    room = max(cfg.max_gaussians - n0, 0)

    n_clone = n_split = 0
    if room > 0:
        small = selected & (scales.max(dim=1).values <= big_thresh)
        large = selected & (scales.max(dim=1).values > big_thresh)

        if small.any():
            idx = small.nonzero().squeeze(-1)[:room]
            n_clone = idx.numel()
            if n_clone:
                _append(
                    opt, m,
                    [m.means.detach()[idx], m.scales.detach()[idx], m.quats.detach()[idx],
                     m.opacities.detach()[idx], m.sh0.detach()[idx], m.shN.detach()[idx]],
                )
                room -= n_clone

        if large.any() and room > 0:
            idx = large.nonzero().squeeze(-1)[: room // 2]
            n_split = idx.numel()
            if n_split:
                sc = torch.exp(m.scales.detach()[idx])
                # Offset each child along the parent's own principal axes.
                noise = torch.randn((n_split, 2, 3), device=dev) * sc.unsqueeze(1)
                base = m.means.detach()[idx].unsqueeze(1) + noise
                shrunk = torch.log(sc / 1.6)
                _append(
                    opt, m,
                    [base.reshape(-1, 3),
                     shrunk.repeat_interleave(2, dim=0),
                     m.quats.detach()[idx].repeat_interleave(2, dim=0),
                     m.opacities.detach()[idx].repeat_interleave(2, dim=0),
                     m.sh0.detach()[idx].repeat_interleave(2, dim=0),
                     m.shN.detach()[idx].repeat_interleave(2, dim=0)],
                )
                # Parents are replaced by their children.
                keep = torch.ones(m.n, dtype=torch.bool, device=dev)
                keep[idx] = False
                _prune(opt, m, keep)

    keep = torch.sigmoid(m.opacities.detach()) > cfg.min_opacity
    keep &= torch.exp(m.scales.detach()).max(dim=1).values < 0.5 * extent
    n_pruned = int((~keep).sum())
    if n_pruned and int(keep.sum()) > 100:
        _prune(opt, m, keep)

    return {"cloned": int(n_clone), "split": int(n_split), "pruned": n_pruned,
            "n_before": n0, "n_after": m.n}


@torch.no_grad()
def reset_opacity(opt, m: GaussianModel, value: float = 0.01):
    """Periodically knock opacities down so floaters must re-earn their weight."""
    new_op = torch.logit(
        torch.full_like(torch.sigmoid(m.opacities.detach()), value)
    ).requires_grad_(True)
    tensors = [m.means.detach().requires_grad_(True),
               m.scales.detach().requires_grad_(True),
               m.quats.detach().requires_grad_(True),
               new_op,
               m.sh0.detach().requires_grad_(True),
               m.shN.detach().requires_grad_(True)]
    _reindex_adam(opt, m, tensors)


# --------------------------------------------------------------------------
# COLMAP ingest
# --------------------------------------------------------------------------


@dataclass
class CameraView:
    name: str
    viewmat: np.ndarray  # (4,4) world-to-camera
    K: np.ndarray        # (3,3)
    width: int
    height: int
    image_path: Path


def load_colmap(model_dir: str | Path, image_dir: str | Path):
    """Read a COLMAP sparse model into views plus a seed point cloud."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(model_dir))
    image_dir = Path(image_dir)

    views: list[CameraView] = []
    for img in rec.images.values():
        cam = rec.cameras[img.camera_id]
        # pycolmap 4.x exposes the pose as cam_from_world (a Rigid3d).
        rigid = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        R = rigid.rotation.matrix()
        t = np.asarray(rigid.translation).reshape(3)
        vm = np.eye(4, dtype=np.float64)
        vm[:3, :3] = R
        vm[:3, 3] = t

        K = cam.calibration_matrix()
        p = image_dir / img.name
        if not p.exists():
            continue
        views.append(
            CameraView(
                name=img.name, viewmat=vm, K=np.asarray(K, dtype=np.float64),
                width=cam.width, height=cam.height, image_path=p,
            )
        )

    views.sort(key=lambda v: v.name)

    xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    rgb = np.array([p.color for p in rec.points3D.values()], dtype=np.float32) / 255.0
    return views, xyz, rgb


def scene_extent(views: list[CameraView]) -> float:
    """Radius of the camera constellation; sets means-LR and densify scale."""
    centers = []
    for v in views:
        R, t = v.viewmat[:3, :3], v.viewmat[:3, 3]
        centers.append(-R.T @ t)
    c = np.stack(centers, axis=0)
    return float(np.linalg.norm(c - c.mean(axis=0), axis=1).max())


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def train_splat(
    model_dir: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    cfg: SplatConfig | None = None,
    log_every: int = 250,
) -> dict:
    """Fit gaussians to a COLMAP reconstruction and export a viewer-ready PLY."""
    import cv2
    from gsplat import rasterization

    cfg = cfg or SplatConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    views, xyz, rgb = load_colmap(model_dir, image_dir)
    if len(views) < 4 or len(xyz) < 100:
        raise RuntimeError(f"model too sparse: {len(views)} views, {len(xyz)} points")

    extent = scene_extent(views)
    cfg.scene_scale = extent
    m = init_from_points(xyz, rgb, cfg)
    opt = make_optimizer(m, cfg, extent)
    dev = torch.device(cfg.device)

    # Images stay on CPU; only the active view is uploaded each iteration.
    cache: dict[str, torch.Tensor] = {}

    def get_image(v: CameraView) -> torch.Tensor:
        if v.name not in cache:
            bgr = cv2.imread(str(v.image_path))
            rgb_ = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            cache[v.name] = torch.from_numpy(rgb_).permute(2, 0, 1)
        return cache[v.name]

    viewmats = torch.tensor(
        np.stack([v.viewmat for v in views]), dtype=torch.float32, device=dev
    )
    Ks = torch.tensor(np.stack([v.K for v in views]), dtype=torch.float32, device=dev)

    grad_accum = torch.zeros(m.n, device=dev)
    grad_count = torch.zeros(m.n, device=dev)
    history: list[dict] = []
    order = np.random.permutation(len(views))
    ptr = 0

    for it in range(1, cfg.iters + 1):
        if ptr >= len(order):
            order = np.random.permutation(len(views))
            ptr = 0
        idx = int(order[ptr]); ptr += 1
        v = views[idx]

        gt = get_image(v).to(dev, non_blocking=True)
        # SH degree ramps in, so low-frequency colour is fit before view-dependence.
        active_sh = min(cfg.sh_degree, it // max(cfg.warmup, 1))
        colors = torch.cat([m.sh0, m.shN], dim=1)

        render, alpha, info = rasterization(
            means=m.means,
            quats=_quat_norm(m.quats),
            scales=torch.exp(m.scales),
            opacities=torch.sigmoid(m.opacities),
            colors=colors,
            viewmats=viewmats[idx : idx + 1],
            Ks=Ks[idx : idx + 1],
            width=v.width,
            height=v.height,
            sh_degree=active_sh,
            packed=True,
            absgrad=True,
        )
        pred = render[0].permute(2, 0, 1).clamp(0, 1)

        l1 = F.l1_loss(pred, gt)
        loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1 - _ssim(pred, gt))

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Adaptive density control: accumulate view-space positional gradient,
        # which is the 3DGS signal for "this region is under-reconstructed".
        if it < cfg.densify_until:
            with torch.no_grad():
                g = info.get("means2d")
                if g is not None and getattr(g, "absgrad", None) is not None:
                    gn = g.absgrad.norm(dim=-1).squeeze(0)
                    gid = info.get("gaussian_ids")
                    if gid is not None and grad_accum.shape[0] == m.n:
                        grad_accum.index_add_(0, gid, gn)
                        grad_count.index_add_(0, gid, torch.ones_like(gn))

        opt.step()

        if (
            it < cfg.densify_until
            and it > cfg.warmup
            and it % cfg.densify_every == 0
        ):
            gmean = grad_accum / grad_count.clamp_min(1.0)
            stats = densify_and_prune(opt, m, gmean, cfg, extent)
            grad_accum = torch.zeros(m.n, device=dev)
            grad_count = torch.zeros(m.n, device=dev)
            if it % (cfg.densify_every * 5) == 0:
                print(json.dumps({"iter": it, **stats}), flush=True)

        if it % cfg.opacity_reset_every == 0 and it < cfg.densify_until:
            reset_opacity(opt, m)
            grad_accum = torch.zeros(m.n, device=dev)
            grad_count = torch.zeros(m.n, device=dev)

        if it % log_every == 0 or it == 1:
            entry = {
                "iter": it,
                "loss": round(loss.item(), 5),
                "l1": round(l1.item(), 5),
                "psnr": round(_psnr(pred, gt), 3),
                "n_gaussians": m.n,
                "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if dev.type == "cuda" else 0.0,
            }
            history.append(entry)
            print(json.dumps(entry), flush=True)

    ply = export_ply(m, out_dir / "splat.ply")
    report = {
        "ok": True,
        "views": len(views),
        "seed_points": int(xyz.shape[0]),
        "final_gaussians": m.n,
        "scene_extent": round(extent, 3),
        "iters": cfg.iters,
        "final_psnr": history[-1]["psnr"] if history else None,
        "peak_vram_gb": history[-1]["vram_gb"] if history else None,
        "ply": ply,
        "history": history,
    }
    (out_dir / "splat_report.json").write_text(json.dumps(report, indent=2))
    return report
