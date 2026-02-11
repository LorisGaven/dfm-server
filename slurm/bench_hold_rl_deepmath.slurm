#!/bin/bash
#SBATCH --job-name=bench-hold_rl_deepmath
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/imi/uej51wk/slurm_logs/bench-hold_rl_deepmath-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/imi/uej51wk/slurm_logs/bench-hold_rl_deepmath-%j.err
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --account=imi@h100

set -eo pipefail

module purge
module load arch/h100
module load python/3.12.2
module load cuda/12.8.0
conda activate dfm-server

CKPT_DIR=$SCRATCH/dfm_generalization/hold_rl_deepmath
PORT=8000
RESULTS="$CKPT_DIR/benchmark_results.json"

echo "=== Checkpoint: $CKPT_DIR ==="
echo "=== Starting server on port $PORT ==="

python -m dfm_server.server \
    --checkpoint "$CKPT_DIR/ckpt.pt" \
    --embeddings "/lustre/fsn1/projects/rech/imi/uej51wk/embeddings.pt" \
    --port $PORT &
SERVER_PID=$!

# Wait for server to be ready
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "=== Server ready (waited ${i}s) ==="
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: Server not ready after 60s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo "=== Running benchmark ==="

python benchmark.py \
    --data "$CKPT_DIR/val.jsonl" \
    --server-url http://localhost:$PORT \
    --output-json "$RESULTS" \
    --max-learners 64

echo "=== Benchmark done, results at $RESULTS ==="

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null || true
