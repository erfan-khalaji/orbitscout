#!/usr/bin/env python3
"""COLMAP SfM for a drone-orbit frame folder -> sparse model for Gaussian splatting.
Pure pip / pycolmap 4.x. CPU-only SIFT (the PyPI wheel has no CUDA).
"""
import argparse, shutil, sys, time
from pathlib import Path
import pycolmap


def dji_focal_px(width_px: int, equiv_mm: float = 24.0) -> float:
    """35mm-equivalent focal -> pixels, assuming the frame spans the full sensor width."""
    return equiv_mm / 36.0 * width_px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("colmap_out"))
    ap.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    ap.add_argument("--equiv-mm", type=float, default=24.0)
    ap.add_argument("--camera-model", default="OPENCV",
                    help="OPENCV/SIMPLE_RADIAL model real distortion; PINHOLE if already rectified")
    ap.add_argument("--max-image-size", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=8, help="cap to bound CPU-SIFT RAM")
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--undistort", action="store_true",
                    help="also emit a PINHOLE undistorted copy (what 3DGS wants)")
    a = ap.parse_args()

    frames = sorted(p for p in a.images.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        sys.exit(f"no images in {a.images}")
    from PIL import Image
    W, H = Image.open(frames[0]).size
    f_px = dji_focal_px(W, a.equiv_mm)
    print(f"{len(frames)} frames @ {W}x{H} -> focal prior {f_px:.1f}px ({a.equiv_mm}mm equiv)")

    a.out.mkdir(parents=True, exist_ok=True)
    db = a.out / "database.db"
    if db.exists():
        db.unlink()
    sparse = a.out / "sparse"
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()

    # ---- intrinsics prior: one shared camera for the whole orbit ----
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = a.camera_model
    if a.camera_model == "OPENCV":            # fx, fy, cx, cy, k1, k2, p1, p2
        reader.camera_params = f"{f_px},{f_px},{W/2},{H/2},0,0,0,0"
    elif a.camera_model == "SIMPLE_RADIAL":   # f, cx, cy, k
        reader.camera_params = f"{f_px},{W/2},{H/2},0"
    elif a.camera_model == "PINHOLE":         # fx, fy, cx, cy
        reader.camera_params = f"{f_px},{f_px},{W/2},{H/2}"
    elif a.camera_model == "SIMPLE_PINHOLE":  # f, cx, cy
        reader.camera_params = f"{f_px},{W/2},{H/2}"

    ext = pycolmap.FeatureExtractionOptions()
    ext.use_gpu = False                        # wheel is CPU-only; True would raise
    ext.num_threads = a.threads
    ext.max_image_size = a.max_image_size
    ext.sift.max_num_features = 8192
    ext.sift.estimate_affine_shape = False
    ext.sift.domain_size_pooling = False

    t = time.time()
    pycolmap.extract_features(
        database_path=db, image_path=a.images,
        camera_mode=pycolmap.CameraMode.SINGLE,     # one camera shared by all frames
        reader_options=reader, extraction_options=ext,
        device=pycolmap.Device.cpu,
    )
    print(f"[1/3] extract {time.time()-t:.1f}s")

    match = pycolmap.FeatureMatchingOptions()
    match.use_gpu = False
    match.num_threads = a.threads
    t = time.time()
    if a.matcher == "exhaustive":
        pycolmap.match_exhaustive(db, matching_options=match,
                                  pairing_options=pycolmap.ExhaustivePairingOptions(),
                                  device=pycolmap.Device.cpu)
    else:
        pair = pycolmap.SequentialPairingOptions()
        pair.overlap = a.overlap
        pair.quadratic_overlap = True
        pair.loop_detection = False      # True needs a downloaded FAISS vocab tree
        pycolmap.match_sequential(db, matching_options=match, pairing_options=pair,
                                  device=pycolmap.Device.cpu)
    print(f"[2/3] match ({a.matcher}) {time.time()-t:.1f}s")

    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = True       # let BA correct the equiv-mm guess
    opts.ba_refine_principal_point = False   # keep pp at center; safer on small sets
    opts.ba_refine_extra_params = True
    opts.min_num_matches = 15
    t = time.time()
    recs = pycolmap.incremental_mapping(database_path=db, image_path=a.images,
                                        output_path=sparse, options=opts)
    print(f"[3/3] map {time.time()-t:.1f}s -> {len(recs)} model(s)")
    if not recs:
        sys.exit("no model reconstructed")

    best_id, best = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    print(best.summary())
    print(f"registered {best.num_reg_images()}/{len(frames)} frames")
    cam = list(best.cameras.values())[0]
    print(f"camera: {cam.model.name} {cam.params}")
    model_dir = sparse / str(best_id)
    print("model dir:", model_dir, sorted(p.name for p in model_dir.iterdir()))

    if a.undistort:
        und = a.out / "undistorted"
        shutil.rmtree(und, ignore_errors=True)
        pycolmap.undistort_images(output_path=und, input_path=model_dir,
                                  image_path=a.images, output_type="COLMAP")
        # undistort_images writes a flat sparse/; 3DGS hardcodes sparse/0
        flat = und / "sparse"
        if (flat / "cameras.bin").exists():
            nested = flat / "0"
            nested.mkdir(exist_ok=True)
            for f in list(flat.glob("*.bin")) + list(flat.glob("*.txt")):
                shutil.move(str(f), str(nested / f.name))
        print("undistorted (PINHOLE) dataset ->", und)
        print("  feed 3DGS with:  python train.py -s", und)


if __name__ == "__main__":
    main()
