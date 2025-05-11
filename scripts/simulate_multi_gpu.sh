#!/usr/bin/env bash
// filepath: /home/azureuser/ccchow/prime/scripts/simulate_multi_node_diloco.sh
set -Eeuo pipefail                           # strict‐mode
shopt -s lastpipe                            # propagate exit codes through pipes

###############################################################################
# simulate many torchrun nodes on ONE physical GPU (or CPU if CUDA is absent)
#   usage: ./simulate_multi_node_diloco.sh <WORLD_SIZE> <GPUS_PER_RANK>  \
#            src/zeroband/train.py @configs/debug/normal.toml
###############################################################################

# ---------- helpers ----------------------------------------------------------
msg()   { printf '\e[1;34m[simulate] %s\e[0m\n' "$*"; }
die()   { printf '\e[1;31m[error] %s\e[0m\n' "$*" >&2; exit 1; }

check_bin() { command -v "$1" >/dev/null || die "missing binary: $1"; }

get_cuda_devices() {
    local num_gpu=$1 idx=$2
    local available=${GPU_COUNT:-0}

    # by default pin everybody to GPU0 (single‑GPU debug)
    if (( available <= 1 )); then
        echo 0; return
    fi

    local start=$(( num_gpu * idx ))
    local end=$(( start + num_gpu - 1 ))
    seq -s, $start $end
}

# ---------- prerequisites ----------------------------------------------------
check_bin torchrun
check_bin uv

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
msg "Detected $GPU_COUNT visible GPU(s)"

(( $# >= 2 )) || die "Usage: $0 <WORLD_SIZE> <GPUS_PER_RANK> [python‑args …]"

WORLD_SIZE=$1
GPUS_PER_RANK=$2
shift 2                                           # $@   -> only python args

# ---------- global env -------------------------------------------------------
export GLOBAL_ADDR=localhost
export GLOBAL_PORT=${GLOBAL_PORT:-5565}
export GLOBAL_WORLD_SIZE=$WORLD_SIZE
export BASE_PORT=${BASE_PORT:-10001}
export GLOO_SOCKET_IFNAME=lo                     # localhost only

LOG_DIR=$(mktemp -d ./logs/run_$(date +%s)_XXXX)
msg "Logs will be written to $LOG_DIR"

# ---------- cleanup ----------------------------------------------------------
child_pids=()
tail_pid=

cleanup() {
    printf '\n'; msg "cleaning up …"
    for pid in "${child_pids[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    [[ -n ${tail_pid:-} ]] && kill -TERM "$tail_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT                       # also on normal exit

# ---------- launch ranks -----------------------------------------------------
for (( i=0; i<WORLD_SIZE; ++i )); do
    log="$LOG_DIR/rank_$i.log"
    : > "$log"
    CUDA_VISIBLE_DEVICES=$(get_cuda_devices "$GPUS_PER_RANK" "$i") \
    WANDB_MODE=$([[ $i == 0 ]] && echo online || echo offline)       \
    GLOBAL_UNIQUE_ID=$i GLOBAL_RANK=$i                              \
    uv run torchrun --nproc_per_node="$GPUS_PER_RANK"               \
                    --node-rank 0                                   \
                    --rdzv-endpoint "localhost:$((BASE_PORT+i))"    \
                    --nnodes 1                                      \
                    "$@"                                            \
                    --data.data_rank "$i"                           \
                    --data.data_world_size "$WORLD_SIZE"            \
                    &> "$log" &
    child_pids+=("$!")
    msg "spawned rank $i (PID ${child_pids[-1]}) -> $log"
done

# live log of rank‑0
tail -Fn0 "$LOG_DIR/rank_0.log" &
tail_pid=$!

# ---------- wait -------------------------------------------------------------
wait -n "${child_pids[@]}"        # exit as soon as ANY rank dies