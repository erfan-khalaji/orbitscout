"""orbitscout command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_scout(args: argparse.Namespace) -> int:
    from .scout import scan

    rows = scan(args.roots, target_hz=args.hz)
    if args.json:
        print(json.dumps([r.as_dict() for r in rows], indent=2))
        return 0

    if not rows:
        print("no telemetry (.SRT) found under the given roots", file=sys.stderr)
        return 1

    hdr = f"{'score':>5} {'iso':>5} {'sweep':>6} {'dur':>6} {'rad':>5} {'alt':>5}  {'when':16}  {'clip':30} verdict"
    print(hdr)
    print("-" * len(hdr))
    for r in rows[: args.top]:
        print(
            f"{r.score:5.2f} {r.isotropy:5.2f} {r.sweep_deg:6.0f} {r.duration_s:6.0f} "
            f"{r.radius_m:5.0f} {r.alt_m:5.0f}  {r.when:16}  {r.name[:30]:30} {r.verdict}"
        )
    print(f"\n{len(rows)} flights scored.")
    print(
        "\nnote: sweep near 360 with low isotropy is an out-and-back dolly, "
        "not an orbit -- isotropy is the deciding column."
    )
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    from .extract import extract

    rep = extract(
        args.video,
        args.out,
        target=args.target,
        oversample=args.oversample,
        long_edge=args.long_edge,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(rep.__dict__, indent=2))
    return 0


def _cmd_depth(args: argparse.Namespace) -> int:
    from .depth import run_depth

    rep = run_depth(
        args.video,
        args.out,
        encoder=args.encoder,
        max_frames=args.max_frames,
        long_edge=args.long_edge,
    )
    print(json.dumps(rep, indent=2))
    return 0


def _cmd_parallax(args: argparse.Namespace) -> int:
    from .parallax import render_parallax

    out = render_parallax(
        args.frames,
        args.depths,
        args.out,
        amplitude=args.amplitude,
        n_frames=args.n_frames,
        fps=args.fps,
        pattern=args.pattern,
    )
    print(out)
    return 0


def _cmd_sfm(args: argparse.Namespace) -> int:
    from .sfm import run_sfm

    rep = run_sfm(
        args.images,
        args.out,
        overlap=args.overlap,
        quadratic=not args.no_quadratic,
        max_features=args.max_features,
        exhaustive=args.exhaustive,
    )
    print(json.dumps(rep, indent=2))
    return 0 if rep.get("ok") else 1


def _cmd_splat(args: argparse.Namespace) -> int:
    from .splat import SplatConfig, train_splat

    cfg = SplatConfig(iters=args.iters, sh_degree=args.sh_degree)
    rep = train_splat(args.model, args.images, args.out, cfg=cfg)
    print(json.dumps({k: v for k, v in rep.items() if k != "history"}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="orbitscout",
        description="Triage DJI flights for 3D reconstruction, then reconstruct.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scout", help="score flights for reconstruction suitability")
    s.add_argument("roots", nargs="+", help="dirs or .SRT files to scan")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--hz", type=float, default=3.0, help="telemetry resample rate")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_scout)

    e = sub.add_parser("extract", help="extract sharp frames for SfM")
    e.add_argument("video")
    e.add_argument("out")
    e.add_argument("--target", type=int, default=150)
    e.add_argument("--oversample", type=int, default=3)
    e.add_argument("--long-edge", type=int, default=1600)
    e.add_argument("--start", type=float, default=0.0)
    e.add_argument("--end", type=float, default=None)
    e.set_defaults(func=_cmd_extract)

    d = sub.add_parser("depth", help="monocular depth for every frame of a clip")
    d.add_argument("video")
    d.add_argument("out")
    d.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    d.add_argument("--max-frames", type=int, default=0, help="0 = all")
    d.add_argument("--long-edge", type=int, default=1280)
    d.set_defaults(func=_cmd_depth)

    x = sub.add_parser("parallax", help="render a 2.5D parallax move from frames+depth")
    x.add_argument("frames")
    x.add_argument("depths")
    x.add_argument("out")
    x.add_argument("--amplitude", type=float, default=0.035)
    x.add_argument("--n-frames", type=int, default=0, help="0 = match input")
    x.add_argument("--fps", type=float, default=30.0)
    x.add_argument("--pattern", default="orbit", choices=["orbit", "dolly", "sway"])
    x.set_defaults(func=_cmd_parallax)

    f = sub.add_parser("sfm", help="structure-from-motion over extracted frames")
    f.add_argument("images")
    f.add_argument("out")
    f.add_argument("--overlap", type=int, default=12)
    f.add_argument("--no-quadratic", action="store_true")
    f.add_argument("--max-features", type=int, default=8192)
    f.add_argument("--exhaustive", action="store_true")
    f.set_defaults(func=_cmd_sfm)

    g = sub.add_parser("splat", help="train 3D gaussian splatting on a COLMAP model")
    g.add_argument("model", help="COLMAP sparse model dir")
    g.add_argument("images")
    g.add_argument("out")
    g.add_argument("--iters", type=int, default=7000)
    g.add_argument("--sh-degree", type=int, default=3)
    g.set_defaults(func=_cmd_splat)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
