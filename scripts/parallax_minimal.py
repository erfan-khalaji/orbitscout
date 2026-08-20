import math, subprocess, numpy as np, cv2, torch, torch.nn.functional as F
IMG, DEPTH, OUT, N, FPS = "hd_frame.png", "hd_depth.png", "min.mp4", 150, 30
SPREAD, HFOV, RATIO = 0.02, 84.0, 14.0          # near<->far shift as fraction of W

bgr = cv2.imread(IMG); d = cv2.imread(DEPTH, cv2.IMREAD_UNCHANGED).astype(np.float32)
lo, hi = np.percentile(d, (1, 99.5)); disp = np.clip((d - lo) / (hi - lo), 0, 1).astype(np.float32)
H, W = disp.shape; f = 0.5 * W / math.tan(math.radians(HFOV) / 2)
cx, cy = (W - 1) / 2, (H - 1) / 2; zn, zf = 1.0, RATIO
z_of = lambda p: 1.0 / (p * (1 / zn - 1 / zf) + 1 / zf)
zp = float(z_of(np.quantile(disp, 0.55)))                     # pivot depth: held still
t = SPREAD * W / (f * (1 / zn - 1 / zf))                      # orbit radius, metres
crop = 0.07; fo = f / (1 - 2 * crop)                          # zoom so borders never show
band = int(fo * t * (1 / zn - 1 / zf)) + 4                    # max disocclusion width, px

# ---- LDI layer 2: background colour+depth behind every depth edge -----------
k = 2 * band + 1; bgd = cv2.erode(disp, np.ones((k, k), np.uint8))
edge = cv2.morphologyEx(disp, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > .02
occl = (cv2.dilate(edge.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & (disp > bgd + .02)
col = torch.from_numpy(bgr[..., ::-1].copy()).cuda().float().div_(255).permute(2, 0, 1)
w = torch.from_numpy((~occl).astype(np.float32)).cuda()[None]; ker = torch.ones(1, 1, 5, 5).cuda()
for _ in range(80):                                            # background-only diffusion
    if not bool((w <= 0).any()): break
    cs = F.conv2d((col * w)[None], ker.expand(3, 1, 5, 5), padding=2, groups=3)[0]
    ws = F.conv2d(w[None], ker, padding=2)[0]; m = (w <= 0) & (ws > 1e-8)
    col = torch.where(m, cs / ws.clamp(min=1e-8), col); w = torch.where(m, ws / 25, w)

# ---- point cloud = source pixels + background band -------------------------
zt = torch.from_numpy(z_of(disp)).cuda().reshape(-1)
v, u = np.mgrid[0:H, 0:W].astype(np.float32)
ug = torch.from_numpy(u - cx).cuda().reshape(-1); vg = torch.from_numpy(v - cy).cuda().reshape(-1)
P = torch.stack([ug * zt / f, vg * zt / f, zt])
rgb = torch.from_numpy(bgr[..., ::-1].copy()).cuda().float().div_(255).permute(2, 0, 1).reshape(3, -1)
sel = torch.from_numpy(occl).cuda().reshape(-1); zb = z_of(torch.from_numpy(bgd).cuda().reshape(-1)[sel])
P = torch.cat([P, torch.stack([ug[sel] * zb / f, vg[sel] * zb / f, zb])], 1)
rgb = torch.cat([rgb, col.reshape(3, -1)[:, sel]], 1)

pipe = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
    "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
    "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
    "-vf", "noise=alls=2:allf=t", OUT], stdin=subprocess.PIPE)
for n in range(N):
    th = 2 * math.pi * n / N                                   # figure-8 orbit + dolly
    C = np.array([t * math.sin(th), .55 * t * math.sin(2 * th), .04 * (1 - math.cos(th))], np.float32)
    fwd = np.array([0, 0, zp], np.float32) - C; fwd /= np.linalg.norm(fwd)
    r = np.cross(np.array([0, 1, 0], np.float32), fwd); r /= np.linalg.norm(r)
    R = np.stack([r, np.cross(fwd, r), fwd], 1)                # cam->world, look-at pivot
    Q = torch.from_numpy(np.ascontiguousarray(R.T)).cuda() @ (P - torch.from_numpy(C).cuda().view(3, 1))
    zc = Q[2].clamp(min=1e-3); x = fo * Q[0] / zc + cx; y = fo * Q[1] / zc + cy
    x0, y0 = torch.floor(x), torch.floor(y); fx, fy = x - x0, y - y0
    num = torch.zeros(3, H * W).cuda(); den = torch.zeros(H * W).cuda()
    zbuf = torch.full((H * W,), 1e9).cuda(); taps = []
    for dx, dy, wt in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        xi, yi = (x0 + dx).long(), (y0 + dy).long()
        ok = (Q[2] > 1e-3) & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (wt > 1e-3)
        i = (yi.clamp(0, H-1) * W + xi.clamp(0, W-1)) * ok; taps.append((i, wt * ok))
        zbuf.scatter_reduce_(0, i, torch.where(ok, zc, torch.tensor(1e9).cuda()), reduce="amin")
    for i, wt in taps:                                          # soft z-test -> occlusion
        wv = wt * torch.exp(-((zc - zbuf[i]) / (zbuf[i] * .03)).clamp(0, 40))
        den.scatter_add_(0, i, wv)
        for c in range(3): num[c].scatter_add_(0, i, rgb[c] * wv)
    out = (num / den.clamp(min=1e-6)).view(3, H, W)
    wv = (den > 1e-3).view(1, H, W).float()                     # patch 1-2px cracks
    for _ in range(6):
        if not bool((wv <= 0).any()): break
        k3 = torch.ones(1, 1, 3, 3).cuda()
        cs = F.conv2d((out * wv)[None], k3.expand(3,1,3,3), padding=1, groups=3)[0]
        ws = F.conv2d(wv[None], k3, padding=1)[0]; m = (wv <= 0) & (ws > 1e-8)
        out = torch.where(m, cs / ws.clamp(min=1e-8), out); wv = torch.where(m, ws / 9, wv)
    fr = (out.clamp(0,1).permute(1,2,0).flip(-1) * 255).round().byte().cpu().numpy()
    pipe.stdin.write(np.ascontiguousarray(fr).tobytes())
pipe.stdin.close(); pipe.wait(); print("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated()/1e9))
