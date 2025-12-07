set -euo pipefail

# conda activate ffa

# 解析当前脚本所在目录，并固定使用同目录下的 run_attn_bench.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
SCRIPT="${SCRIPT_DIR}/run_attn_bench.py"

BS_LIST=(64 128 256)
SBS_LIST=(64 128 256)
DELTA_LIST=(5.0)

MAX_LENGTH=$(( 32 * 1024 ))

DTYPE=${DTYPE:-fp16}
ITERS=${ITERS:-1000}
WARMUP=${WARMUP:-1000}

KERNEL=${KERNEL:-attn_kernel.attn_kernel_v1109_fused_bsz}


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
        --max-length "${MAX_LENGTH}" \
        ${EXTRA_ARGS}
    done
  done
done
