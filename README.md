# orbitscout

Triage a DJI drone archive for 3D-reconstruction potential from telemetry alone,
then run the reconstruction that the footage can actually support.

Most drone clips will not photogrammetrically reconstruct. Finding that out
costs an afternoon of COLMAP if you discover it by trying. It costs milliseconds
if you read the flight path first.

## Why telemetry triage

Structure-from-motion and Gaussian splatting need the camera to travel *around*
the subject. A clip that dollies past a subject in a straight line gives
parallax along one axis only: you recover a facade, never a volume.

The obvious metric is wrong. Accumulated bearing change about the track centroid
looks like the natural "did it go around?" statistic — but an out-and-back
straight line passes through its own centroid, so the bearing flips ~180° at
each end and totals near 360°, scoring a pure dolly as a perfect orbit.

Real example from the archive this was built against:

| clip | sweep° | isotropy | actual flight |
|---|---|---|---|
| `DJI_20260819210736_0002` | **360** | 0.01 | straight out-and-back dolly |
| `DJI_20241102163450_0027` | 239 | **0.79** | true 209 s orbit |

The clip with a *perfect* 360° sweep score has no orbital coverage at all.
Isotropy of the track's spatial covariance — `sqrt(λmin/λmax)` — disambiguates:
a line has one dominant eigenvalue, a circle has two equal ones. Sweep is kept
only as a supporting signal.

## Install

Requires `ffmpeg` on PATH and an NVIDIA GPU for the depth stage (CPU works, slower).

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .
# PyTorch — cu128+ wheels carry sm_120, required for RTX 50-series Blackwell
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install transformers safetensors accelerate
```

## Use

```bash
# 1. score every flight in an archive, best first
orbitscout scout /path/to/drone/raw

# 2. extract sharp, well-separated frames from the winner
orbitscout extract flight.MP4 out/frames --target 160 --long-edge 1600

# 3. monocular depth over a whole clip, temporally stabilised
orbitscout depth flight.MP4 out/ --encoder vits

# 4. render a 2.5D parallax move from frames + depth
orbitscout parallax out/frames out/depth out/parallax.mp4 --pattern orbit
```

`scout` verdicts:

- `STRONG` — full orbit, feed it to SfM + Gaussian splatting
- `USABLE` — partial arc, expect gaps on the unseen side
- `DOLLY` — straight pass, no orbital parallax; 2.5D depth is the honest ceiling
- `WEAK` — insufficient baseline

## Measured results

Run against a 209 s drone orbit of a bridge pier (DJI Mini 4 Pro, 4K30), on an
RTX 5060 Laptop with 8 GB of VRAM. Every number below came out of this repo.

**Structure from motion** — 160 extracted frames at 1600x900, CPU SIFT:

| | |
|---|---|
| registered | 121 / 160 (75.6%) |
| 3D points | 20,082 |
| mean reprojection error | **0.44 px** |
| mean track length | 16.3 views |
| wall clock | 2.2 min |

**GPS cross-check** — the reconstruction validated against telemetry that was
never an input to the solve:

| | |
|---|---|
| frames aligned | 121 |
| recovered metric scale | 12.247 model-units per metre |
| mean residual | **1.17 m** |
| median / p90 | 1.05 m / 2.15 m |
| over a GPS track of | 104 m |

Vision-only geometry and GPS independently agree to about 1% of track extent.

**Depth + parallax** — 214-frame sunset clip, Depth Anything V2 Small:
temporal flicker 0.0026 after stabilisation; near/far displacement ratio
**14.2x**, rising monotonically with disparity (0.89 -> 1.29 -> 2.54 px across
disparity terciles), which is the signature of real parallax rather than a
global pan.

## Design notes

**Frame extraction** oversamples and keeps the sharpest frame per temporal
bucket (variance of Laplacian). Fixed-interval sampling lands on motion-blurred
frames, and blur poisons feature matching. Decoding is one sequential ffmpeg
pass rather than N random seeks — 4K HEVC seeking costs a keyframe hunt plus
re-decode every time, so hundreds of seeks take minutes where one linear pass
takes seconds.

**Depth** is relative inverse depth, arbitrary and independent per frame, which
flickers over video. Two fixes: percentiles are pooled across the whole clip so
one fixed normalisation applies to every frame, and a forward-backward EMA
removes residual flicker without the phase lag a causal filter would add.

**Parallax** exploits `du = f·tx/Z = k·disparity`: for small virtual baselines
image-space motion is proportional to disparity, so a normalised inverse-depth
map *is* a displacement field. Warping is forward with a z-buffer, because
backward warping needs an inverse field that does not exist across depth
discontinuities. Disocclusions are filled by a depth-aware pull from valid
neighbours.

## Licensing

Code here is MIT. Model weights are pulled at runtime and carry their own terms:
Depth Anything V2 **Small is Apache-2.0**; Base and Large are **CC-BY-NC-4.0**
(free to download, non-commercial use only). Small is the default for that
reason. Nothing in this pipeline requires a paid service.
