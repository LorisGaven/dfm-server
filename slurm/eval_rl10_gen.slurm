#!/bin/bash
#SBATCH --job-name=dfm-eval-rl10-gen
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fswork/projects/rech/imi/uej51wk/slurm_logs/dfm-eval-rl10-gen.out
#SBATCH --error=/lustre/fswork/projects/rech/imi/uej51wk/slurm_logs/dfm-eval-rl10-gen.err
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --account=imi@h100

module purge
module load arch/h100
module load python/3.12.2
module load cuda/12.8.0
conda activate dfm-server

export HF_HUB_OFFLINE=1

# --- Configuration (edit these) ---
CHECKPOINT=${DFM_CHECKPOINT:-$SCRATCH/dfm-training/rl10_gen/ckpt.pt}
TASKS=${DFM_TASKS:-$SCRATCH/tasks.json}
TRAJECTORY=${TRAJECTORY:-$SCRATCH/dfm-test/rl_Qwen2.5-1.5B-Instruct_dapo_bs64_lr5e-6_seed42/history.json}
EVAL_DIR=${EVAL_DIR:-$SCRATCH/dfm-test/rl_eval_Qwen2.5-1.5B-Instruct_dapo}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
RESULTS_DIR=${RESULTS_DIR:-$SCRATCH/dfm-training/rl10_gen}

mkdir -p "$RESULTS_DIR"

# --- Start DFM server ---
DFM_PORT=8020
DFM_LOG="$RESULTS_DIR/server.log"

echo "Starting DFM server (checkpoint=$CHECKPOINT, tasks=$TASKS)..."
dfm-server \
    --checkpoint "$CHECKPOINT" \
    --tasks "$TASKS" \
    --device cuda \
    --port $DFM_PORT \
    > "$DFM_LOG" 2>&1 &
DFM_PID=$!

# Wait for server readiness
echo "Waiting for DFM server on port $DFM_PORT (pid=$DFM_PID)..."
while ! curl -s "http://localhost:$DFM_PORT/health" > /dev/null 2>&1; do
    if ! kill -0 $DFM_PID 2>/dev/null; then
        echo "ERROR: DFM server died during startup. Log:"
        cat "$DFM_LOG"
        exit 1
    fi
    sleep 2
done
echo "DFM server ready"

# Cleanup server on exit
trap 'kill $DFM_PID 2>/dev/null' EXIT

# --- Run evaluation ---
python -m benchmark.eval_rl \
    --trajectory "$TRAJECTORY" \
    --eval-dir "$EVAL_DIR" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --server-url "http://localhost:$DFM_PORT" \
    --grid-step 512 \
    --max-context 4096 \
    --max-horizon 4096 \
    --min-step 100 \
    --predict-batch-size 1024 \
    --save "$RESULTS_DIR"

echo "Evaluation complete. Results in $RESULTS_DIR"
