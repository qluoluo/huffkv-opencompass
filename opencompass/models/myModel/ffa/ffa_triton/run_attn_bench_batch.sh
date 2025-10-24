set -euo pipefail

# conda activate ffa

# 解析当前脚本所在目录，并固定使用同目录下的 run_attn_bench.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
SCRIPT="${SCRIPT_DIR}/run_attn_bench.py"

# 参数集合可按需调整
# BS_LIST=(64 128 256 512)
# SBS_LIST=(64 128 256 512)
# DELTA_LIST=(5.0 10.0)

# BS_LIST=(128 256 512 1024)
# SBS_LIST=(128 256 512)
BS_LIST=(256 512)
SBS_LIST=(256 512)
DELTA_LIST=(5.0)

DTYPE=${DTYPE:-fp16}
ITERS=${ITERS:-1000}
WARMUP=${WARMUP:-1000}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1019_unfused}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1019_fused}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1022_fused_grid1d}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1022_fused_grid2d_ht}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1022_unfused_grid2d_ht}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1023_fused_tk}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1023_fused_tbs4}
# KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1024_onekernel_fullk}
KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1024_flashdecoding}



STEP=${STEP:-1024}
# 可注入额外参数，比如 --no-plot-line
EXTRA_ARGS=${EXTRA_ARGS:-}

# 如需限制 GPU：
# export CUDA_VISIBLE_DEVICES=0

for BS in "${BS_LIST[@]}"; do
  for SBS in "${SBS_LIST[@]}"; do
    # 跳过 SBS 大于 BS 的组合
    if (( SBS > BS )); then
    #   echo "Skip combo: SBS=${SBS} > BS=${BS}"
      continue
    fi
    for DELTA in "${DELTA_LIST[@]}"; do
      echo "Running: BS=$BS SBS=$SBS delta=$DELTA dtype=$DTYPE kernel=$KERNEL"
      ${PY} ${SCRIPT} \
        --dtype "${DTYPE}" \
        --BS "${BS}" \
        --SBS "${SBS}" \
        --delta "${DELTA}" \
        --iters "${ITERS}" \
        --warmup "${WARMUP}" \
        --kernel "${KERNEL}" \
        --step "${STEP}" \
        ${EXTRA_ARGS}
    done
  done
done
