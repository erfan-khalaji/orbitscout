"""Structure-from-motion over extracted frames, via pycolmap.

Runs entirely from a pip wheel -- no system COLMAP, no sudo. The wheel ships
CPU SIFT only, which is slower than a CUDA build but perfectly adequate at the
frame counts this pipeline produces.

Matching strategy matters for orbits. Exhaustive matching is O(n^2) pairs and
wasteful when frames are already time-ordered: frame 0 cannot possibly match
frame 80 on the far side of the subject. Sequential matching with quadratic
overlap links each frame to its neighbours at exponentially increasing stride
(i+1, i+2, i+4, i+8...), which keeps the pair count near-linear while still
providing the long-range links that let an orbit close its loop.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pycolmap


def _component_stats(rec) -> dict:
    """Per-component geometry summary, including angular coverage.

    Angular coverage is the arc the cameras span as seen from the scene
    centroid. Near 360 means the capture genuinely surrounded the subject and
    a synthetic orbit will render cleanly; a handful of degrees means the
    component is effectively forward-facing, and novel views must stay close
    to the original trajectory.
    """
    import numpy as np

    centers = []
    for img in rec.images.values():
        rigid = img.cam_from_world() if callable(getattr(img, "cam_from_world", None)) else img.cam_from_world
        R = np.asarray(rigid.rotation.matrix())
        t = np.asarray(rigid.translation).reshape(3)
        centers.append(-R.T @ t)
    if len(centers) < 3 or not rec.points3D:
        return {"images": len(centers), "points": len(rec.points3D),
                "angular_coverage_deg": 0.0, "mean_distance": 0.0}

    C = np.stack(centers)
    xyz = np.array([p.xyz for p in rec.points3D.values()])
    lo, hi = np.percentile(xyz, [5, 95], axis=0)
    core = xyz[np.all((xyz >= lo) & (xyz <= hi), axis=1)]
    target = core.mean(axis=0) if len(core) > 50 else xyz.mean(axis=0)

    d = C - target
    # Project onto the plane perpendicular to the dominant camera-spread normal.
    # full_matrices=True so Vt is always 3x3: with fewer cameras than dimensions
    # the economy SVD returns too few rows and there is no third basis vector.
    _, _, Vt = np.linalg.svd(C - C.mean(axis=0), full_matrices=True)
    n = Vt[2]
    proj = d - np.outer(d @ n, n)
    e0 = Vt[0] - np.dot(Vt[0], n) * n
    e0 /= max(np.linalg.norm(e0), 1e-9)
    e1 = np.cross(n, e0)
    ang = np.arctan2(proj @ e1, proj @ e0)
    return {
        "images": len(C),
        "points": len(rec.points3D),
        "angular_coverage_deg": round(float(np.degrees(ang.max() - ang.min())), 1),
        "mean_distance": round(float(np.linalg.norm(d, axis=1).mean()), 3),
    }


def run_sfm(
    image_dir: str | Path,
    out_dir: str | Path,
    overlap: int = 12,
    quadratic: bool = True,
    max_features: int = 8192,
    single_camera: bool = True,
    exhaustive: bool = False,
) -> dict:
    """Feature-extract, match and map a frame directory into a COLMAP model."""
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = out_dir / "database.db"
    sparse = out_dir / "sparse"
    if db.exists():
        db.unlink()
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)

    n_images = len(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if n_images < 8:
        raise RuntimeError(f"only {n_images} images in {image_dir}; need at least 8")

    # All frames come from one video, so they share one physical camera and one
    # set of intrinsics. Telling COLMAP that collapses thousands of unknowns
    # into one camera model and markedly stabilises the solve.
    ex_opts = pycolmap.FeatureExtractionOptions()
    ex_opts.sift.max_num_features = max_features
    pycolmap.extract_features(
        database_path=db,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO,
        extraction_options=ex_opts,
    )

    if exhaustive:
        pycolmap.match_exhaustive(database_path=db)
        strategy = "exhaustive"
    else:
        pair = pycolmap.SequentialPairingOptions()
        pair.overlap = overlap
        pair.quadratic_overlap = quadratic
        pycolmap.match_sequential(database_path=db, pairing_options=pair)
        strategy = f"sequential(overlap={overlap}, quadratic={quadratic})"

    recons = pycolmap.incremental_mapping(
        database_path=db, image_path=image_dir, output_path=sparse
    )
    if not recons:
        return {
            "ok": False,
            "reason": "no reconstruction produced -- insufficient parallax or matches",
            "n_images": n_images,
            "match_strategy": strategy,
        }

    # incremental_mapping often returns several disconnected components, and
    # "largest" is not the same as "most useful". On a drone orbit of a distant
    # subject the biggest component is routinely a near-collinear run of frames
    # covering only a few degrees around the scene, while a smaller component
    # holds the part that actually swept around it. Angular coverage is
    # reported per component so that choice can be made deliberately.
    stats = {k: _component_stats(r) for k, r in recons.items()}
    best_id = max(recons, key=lambda k: recons[k].num_reg_images())
    rec = recons[best_id]

    errs = [p.error for p in rec.points3D.values() if p.error >= 0]
    track_lens = [p.track.length() for p in rec.points3D.values()]

    ply = out_dir / "points.ply"
    rec.export_PLY(str(ply))

    report = {
        "ok": True,
        "n_images": n_images,
        "registered": rec.num_reg_images(),
        "registration_rate": round(rec.num_reg_images() / n_images, 3),
        "points3D": len(rec.points3D),
        "mean_reproj_error_px": round(sum(errs) / len(errs), 4) if errs else None,
        "mean_track_length": round(sum(track_lens) / len(track_lens), 2) if track_lens else None,
        "n_components": len(recons),
        "components": {
            str(k): v for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["images"])
        },
        "match_strategy": strategy,
        "model_dir": str(sparse / str(best_id)),
        "ply": str(ply),
    }
    (out_dir / "sfm_report.json").write_text(json.dumps(report, indent=2))
    return report
