#!/usr/bin/env python3
"""
2.5D parallax ("3D photo" / Ken-Burns-3D) video from ONE still + ONE monocular depth map.

Method: 2-layer LDI (source pixels + a background band inpainted by background-only
diffusion) -> chunked forward splat with a soft z-buffer -> look-at orbit/dolly path
-> raw frames piped to ffmpeg.  numpy + opencv + torch only, ~2 GB VRAM at 4K.

  python scripts/parallax25d.py frame.jpg depth.png out.mp4 --frames 150 --spread 0.02
"""
import argparse, math, subprocess, time
import numpy as np, cv2, torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK = 6_000_000                       # points per splat batch -> bounds VRAM


# --------------------------------------------------------------- input prep
def load_pair(img_path, depth_path, max_side, invert=False, clip=(1.0, 99.5)):
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    d = (np.load(depth_path) if d is None else d).astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    s = min(1.0, max_side / max(bgr.shape[:2]))
    if s < 1.0:
        bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if d.shape[:2] != bgr.shape[:2]:
        d = cv2.resize(d, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    if invert:
        d = -d
    lo, hi = np.percentile(d, clip)      # robust normalise; ignores depth outliers
    return bgr, np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def to_z(disp, zn, zf):
    """relative disparity in [0,1] (1 = nearest) -> depth, linear in disparity."""
    return 1.0 / (disp * (1.0 / zn - 1.0 / zf) + 1.0 / zf)


def background_layer(bgr, disp, band, tau=0.02, iters=80):
    """LDI layer 2: colour + depth of what hides behind every depth discontinuity.
    Seeded ONLY by background pixels, so foreground colour can never bleed in."""
    k = 2 * band + 1
    bgd = cv2.erode(disp, np.ones((k, k), np.uint8))                 # bg disparity
    edge = cv2.morphologyEx(disp, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > tau
    occl = (cv2.dilate(edge.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) \
           & (disp > bgd + tau)                                      # near side of edge
    col = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(DEV).float() \
               .div_(255).permute(2, 0, 1)
    w = torch.from_numpy((~occl).astype(np.float32)).to(DEV)[None]
    ker = torch.ones(1, 1, 5, 5, device=DEV)
    for _ in range(iters):
        if not bool((w <= 0).any()):
            break
        cs = F.conv2d((col * w)[None], ker.expand(3, 1, 5, 5), padding=2, groups=3)[0]
        ws = F.conv2d(w[None], ker, padding=2)[0]
        m = (w <= 0) & (ws > 1e-8)
        col = torch.where(m, cs / ws.clamp(min=1e-8), col)
        w = torch.where(m, ws / 25.0, w)
    return col, torch.from_numpy(bgd).to(DEV), torch.from_numpy(occl).to(DEV)


# -------------------------------------------------------------- camera path
def pose(s, ax, ay, az, zp):
    """s in [0,1): figure-8 orbit + dolly about pivot P=(0,0,zp), always looking at P.
    Returns R (cam->world) and C (camera centre). Period 1 -> seamless loop."""
    th = 2 * math.pi * s
    C = np.array([ax * math.sin(th), ay * math.sin(2 * th),
                  0.5 * az * (1 - math.cos(th))], np.float32)
    fwd = np.array([0, 0, zp], np.float32) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.array([0, 1, 0], np.float32), fwd)     # +y is DOWN (image coords)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd], 1).astype(np.float32), C


def auto_crop(f, W, H, zn, zf, zp, ax, ay, az, n=64):
    """Zoom needed so the warped source always covers the output rect (no black edges).
    Displacement scales with focal, so the crop fraction c solves 2c^2 - c + a = 0."""
    A = 0.0
    for i in range(n):
        R, C = pose(i / n, ax, ay, az, zp)
        for z in (zn, zf):
            for ux, uy in ((-W / 2, -H / 2), (W / 2, -H / 2), (-W / 2, H / 2), (W / 2, H / 2)):
                q = R.T @ (np.array([ux * z / f, uy * z / f, z], np.float32) - C)
                A = max(A, abs(f * q[0] / q[2] - ux) / W, abs(f * q[1] / q[2] - uy) / H)
    A = min(1.02 * A + 2.0 / W, 0.115)          # 2% + 2px safety for edge taps
    return (1 - math.sqrt(max(1 - 8 * A, 0.0))) / 4


# ---------------------------------------------------------------- splatting
def splat(P, rgb, R, C, f, cx, cy, H, W, soft=0.03):
    """Forward warp with a soft z-buffer, in point chunks to bound VRAM."""
    Rt = torch.from_numpy(np.ascontiguousarray(R.T)).to(DEV)
    Ct = torch.from_numpy(C).to(DEV).view(3, 1)
    num = torch.zeros(3, H * W, device=DEV)
    den = torch.zeros(H * W, device=DEV)
    zbuf = torch.full((H * W,), 1e9, device=DEV)
    big = torch.tensor(1e9, device=DEV)

    def taps(sl):
        Q = Rt @ (P[:, sl] - Ct)
        z = Q[2].clamp(min=1e-3)
        x, y = f * Q[0] / z + cx, f * Q[1] / z + cy
        x0, y0 = torch.floor(x), torch.floor(y)
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            ok = (Q[2] > 1e-3) & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (w > 1e-3)
            yield (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)) * ok, w * ok, z

    for a in range(0, P.shape[1], CHUNK):                     # pass 1: build z-buffer
        for i, w, z in taps(slice(a, a + CHUNK)):
            zbuf.scatter_reduce_(0, i, torch.where(w > 0, z, big), reduce="amin",
                                 include_self=True)
    for a in range(0, P.shape[1], CHUNK):                     # pass 2: soft accumulate
        sl = slice(a, a + CHUNK)
        for i, w, z in taps(sl):
            wv = w * torch.exp(-((z - zbuf[i]) / (zbuf[i] * soft)).clamp(0.0, 40.0))
            den.scatter_add_(0, i, wv)
            for c in range(3):
                num[c].scatter_add_(0, i, rgb[c, sl] * wv)
    return (num / den.clamp(min=1e-6)).view(3, H, W), (den > 1e-3).view(1, H, W)


def patch_cracks(col, valid, iters=6):
    """Only 1-2 px sampling cracks should survive the LDI; average valid neighbours."""
    w = valid.float()
    ker = torch.ones(1, 1, 3, 3, device=DEV)
    for _ in range(iters):
        if not bool((w <= 0).any()):
            break
        cs = F.conv2d((col * w)[None], ker.expand(3, 1, 3, 3), padding=1, groups=3)[0]
        ws = F.conv2d(w[None], ker, padding=1)[0]
        m = (w <= 0) & (ws > 1e-8)
        col = torch.where(m, cs / ws.clamp(min=1e-8), col)
        w = torch.where(m, ws / 9.0, w)
    return col.clamp(0, 1)


# ------------------------------------------------------------------- driver
def render(a):
    bgr, disp = load_pair(a.image, a.depth, a.max_side, a.invert)
    H, W = disp.shape
    f = 0.5 * W / math.tan(math.radians(a.hfov) / 2)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    zn, zf = 1.0, float(a.depth_ratio)
    zp = float(to_z(np.quantile(disp, a.pivot_q), zn, zf))    # pivot on real content
    t = a.spread * W / (f * (1 / zn - 1 / zf))                # |t| for wanted near<->far spread
    ax, ay, az = t, a.vert * t, min(a.dolly * zp, 0.08 * zn)
    crop = auto_crop(f, W, H, zn, zf, zp, ax, ay, az)
    f_out = f / (1 - 2 * crop)
    band = int(f_out * math.hypot(ax, ay) * (1 / zn - 1 / zf)) + 4
    print(f"{W}x{H} f={f:.0f}px pivot_z={zp:.2f} |t|={t:.4f} dolly={az:.4f} "
          f"crop={100*crop:.2f}% band={band}px")

    z = torch.from_numpy(to_z(disp, zn, zf)).to(DEV).reshape(-1)
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    ug = torch.from_numpy(u - cx).to(DEV).reshape(-1)
    vg = torch.from_numpy(v - cy).to(DEV).reshape(-1)
    P = torch.stack([ug * z / f, vg * z / f, z])
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1])).to(DEV).float() \
               .div_(255).permute(2, 0, 1).reshape(3, -1)
    if not a.no_ldi:
        bcol, bdisp, occl = background_layer(bgr, disp, band)
        sel = occl.reshape(-1)
        zb = to_z(bdisp.reshape(-1)[sel], zn, zf)
        P = torch.cat([P, torch.stack([ug[sel] * zb / f, vg[sel] * zb / f, zb])], 1)
        rgb = torch.cat([rgb, bcol.reshape(3, -1)[:, sel]], 1)
        print(f"  LDI layer-2 points: {int(sel.sum()):,} ({100*sel.float().mean():.2f}%)")
        del bcol, bdisp, occl
    del z, ug, vg
    torch.cuda.empty_cache()

    pix = "yuv420p10le" if a.tenbit else "yuv420p"
    pipe = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", "-an", "-c:v", "libx264",
         "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", pix,
         "-vf", "noise=alls=2:allf=t", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE)
    for n in range(a.frames):
        R, C = pose(n / a.frames, ax, ay, az, zp)
        col, valid = splat(P, rgb, R, C, f_out, cx, cy, H, W)
        fr = (patch_cracks(col, valid).permute(1, 2, 0).flip(-1) * 255).round().byte()
        pipe.stdin.write(np.ascontiguousarray(fr.cpu().numpy()).tobytes())
    pipe.stdin.close(); pipe.wait()
    print("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image"); p.add_argument("depth"); p.add_argument("out")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--hfov", type=float, default=84.0, help="camera horizontal FOV, deg")
    p.add_argument("--spread", type=float, default=0.020,
                   help="near<->far relative shift as a fraction of width (0.015-0.03)")
    p.add_argument("--vert", type=float, default=0.55, help="vertical amplitude / lateral")
    p.add_argument("--dolly", type=float, default=0.06, help="dolly-in as fraction of pivot z")
    p.add_argument("--depth-ratio", type=float, default=14.0, help="z_far / z_near")
    p.add_argument("--pivot-q", type=float, default=0.55, help="disparity quantile held still")
    p.add_argument("--max-side", type=int, default=3840)
    p.add_argument("--crf", type=int, default=16)
    p.add_argument("--tenbit", action="store_true", help="10-bit encode (sunset banding)")
    p.add_argument("--invert", action="store_true", help="depth map is metric (near = small)")
    p.add_argument("--no-ldi", action="store_true")
    t0 = time.time()
    render(p.parse_args())
    print("elapsed %.1fs" % (time.time() - t0))
