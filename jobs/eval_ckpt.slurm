#!/bin/bash
#SBATCH --job-name=eval_ckpt
#SBATCH --account=kmarino
#SBATCH --partition=kmarino-gpu-grn
#SBATCH --qos=kmarino-gpu-grn
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1470293/projects/pusht/logs/%x-%j.out

# Evaluate checkpoints of one training run in gym-pusht, one after another on a single GPU.
# Usage: sbatch eval_ckpt.slurm <run_name> <n_episodes> <step> [<step> ...]
#   e.g. sbatch eval_ckpt.slurm dp_run1 500 100000 200000
# Same eval settings as eval_official.slurm (seed 1000, batch 50, sync envs), so every run is scored
# on the same 500 starting states as the official baseline. eval_info.json is copied to home.

RUN=${1:?run name}
N=${2:?n_episodes}
shift 2
BATCH=$(( N < 50 ? N : 50 ))

module load miniforge3/25.11.0
eval "$(conda shell.bash hook)"
conda activate ~/envs/lerobot

export HF_HOME=/scratch/general/vast/u1470293/pusht/hf_cache
export HF_HUB_OFFLINE=1
export SDL_VIDEODRIVER=dummy
SCRATCH=/scratch/general/vast/u1470293/pusht
RESULTS=~/projects/pusht/results/$RUN
mkdir -p $RESULTS

STATUS=0
for STEP in "$@"; do
  CKPT=$SCRATCH/outputs/train/$RUN/checkpoints/$(printf "%06d" $STEP)/pretrained_model
  OUT=$SCRATCH/outputs/eval/${RUN}_${STEP}_${N}ep_$SLURM_JOB_ID
  echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date): $RUN step $STEP, $N episodes"

  lerobot-eval \
    --policy.path=$CKPT \
    --policy.device=cuda \
    --env.type=pusht \
    --eval.n_episodes=$N \
    --eval.batch_size=$BATCH \
    --eval.use_async_envs=false \
    --output_dir=$OUT
  RC=$?

  if [ $RC -eq 0 ]; then
    cp $OUT/eval_info.json $RESULTS/eval_${N}ep_step${STEP}.json
    echo "=== step $STEP done, copied to $RESULTS/eval_${N}ep_step${STEP}.json"
  else
    echo "=== step $STEP FAILED with exit code $RC"
    STATUS=$RC
  fi
done

echo "Exit code $STATUS at $(date)"
exit $STATUS
