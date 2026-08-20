#!/usr/bin/env bash
# Wait until the GPU has enough free VRAM, then train. Laptop GPUs are easily
# contended; starting into a spike wastes an entire run on an OOM at iteration 0.
set -u
NEED_MB=${NEED_MB:-5000}
cd "$(dirname "$0")/.."
source .venv/bin/activate
CU=$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13
OUT=/media/nullframebit/663fdfa6-43ae-4b31-aba4-5b39bf25a9131/drone/processed

for i in $(seq 1 240); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$NEED_MB" ]; then echo "GPU free=${free}MB, starting"; break; fi
  echo "waiting for GPU (free=${free}MB, need ${NEED_MB}MB)"; sleep 20
done

exec env CUDA_HOME=$CU PATH=$CU/bin:$PATH TORCH_CUDA_ARCH_LIST=12.0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -c "
import sys,json; sys.path.insert(0,'src')
from dronecv.splat import SplatConfig, train_splat
cfg=SplatConfig(iters=7000, warmup=500, densify_until=5000, densify_every=100,
                opacity_reset_every=3000, max_gaussians=600000)
r=train_splat('$OUT/bridge_orbit/sfm/sparse/3','$OUT/bridge_orbit/frames',
              '$OUT/bridge_orbit/splat', cfg=cfg, log_every=250)
print(json.dumps({k:v for k,v in r.items() if k!='history'},indent=2))
"
