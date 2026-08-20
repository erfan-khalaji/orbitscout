#!/usr/bin/env bash
# Make gsplat's CUDA extensions build on RTX 50-series (Blackwell, sm_120)
# with NO sudo and NO system CUDA install.
#
# gsplat JIT-compiles ~20 CUDA kernels on first use, which needs a real nvcc.
# The whole toolchain can come from PyPI, but three things bite:
#
#  1. VERSION CONSISTENCY. nvcc, nvvm/cicc, crt and cccl must all be the same
#     release. Mixing them produces two distinct, confusing failures:
#       - "CUDA compiler and CUDA toolkit headers are incompatible"
#         (nvcc version != cccl header version)
#       - "ptxas fatal: Unsupported .version 9.3; current version is '9.0'"
#         (a newer cicc emits PTX that an older ptxas cannot assemble)
#     Torch already depends on a mutually consistent set, so the safe move is
#     to reuse exactly the versions torch pulled in rather than pick your own.
#
#  2. THE MISSING SYMLINK. The wheels ship libcudart.so.13 but no unversioned
#     libcudart.so, so the final link step fails with `cannot find -lcudart`
#     after all 20 kernels have already compiled successfully.
#
#  3. nvvm IS A SEPARATE PACKAGE, named `nvidia-nvvm` (not nvidia-cuda-nvvm).
#     Without it there is no cicc and nvcc cannot compile device code at all.
#
# Never `rm -rf` the nvidia/cu13 directory to "clean up": it is shared by many
# installed packages (cublas, cudart, nvvm...), and deleting it silently breaks
# torch itself while pip still believes everything is installed.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

CU="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"

echo "==> torch CUDA: $(python -c 'import torch;print(torch.version.cuda)')"
echo "==> compute capability: $(python -c 'import torch;print(torch.cuda.get_device_capability(0))')"

# nvcc + the pieces it needs, at whatever versions torch already settled on.
python - <<'PY'
import subprocess, sys
want = ["nvidia-cuda-nvcc", "nvidia-nvvm", "nvidia-cuda-crt", "nvidia-cuda-cccl"]
out = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True).stdout
have = {}
for line in out.splitlines():
    parts = line.split()
    if len(parts) >= 2 and parts[0] in want:
        have[parts[0]] = parts[1]
missing = [w for w in want if w not in have]
if missing:
    print(f"installing missing: {missing}")
    subprocess.run(["uv", "pip", "install", "-q", *missing], check=True)
else:
    print("toolchain packages present:", have)
PY

# The link step needs an unversioned soname.
if [ ! -e "$CU/lib/libcudart.so" ]; then
  ln -sf libcudart.so.13 "$CU/lib/libcudart.so"
  echo "==> created libcudart.so symlink"
fi

for b in nvcc ptxas nvlink; do
  printf '    %-7s %s\n' "$b" "$("$CU/bin/$b" --version 2>&1 | grep -oE 'V[0-9.]+' | tail -1)"
done
[ -x "$CU/nvvm/bin/cicc" ] && echo "    cicc    present" || { echo "    cicc    MISSING"; exit 1; }

echo "==> building gsplat kernels for sm_120 (one-off, ~1 min)"
CUDA_HOME="$CU" PATH="$CU/bin:$PATH" TORCH_CUDA_ARCH_LIST="12.0" python - <<'PY'
import torch
from gsplat import rasterization
N, dev = 1000, "cuda"
q = torch.zeros(N, 4, device=dev); q[:, 0] = 1
K = torch.tensor([[[300., 0, 160.], [0, 300., 120.], [0, 0, 1.]]], device=dev)
vm = torch.eye(4, device=dev).unsqueeze(0)
m = torch.randn(N, 3, device=dev, requires_grad=True)
out, _, _ = rasterization(
    means=m, quats=q, scales=torch.full((N, 3), 0.05, device=dev),
    opacities=torch.full((N,), 0.5, device=dev), colors=torch.rand(N, 3, device=dev),
    viewmats=vm, Ks=K, width=320, height=240)
out.sum().backward(); torch.cuda.synchronize()
print(f"    forward OK {tuple(out.shape)}, backward OK grad={float(m.grad.norm()):.1f}")
PY

cat <<MSG

==> done. Export these for any process that trains splats:

    export CUDA_HOME=$CU
    export PATH=\$CUDA_HOME/bin:\$PATH
    export TORCH_CUDA_ARCH_LIST=12.0
MSG
