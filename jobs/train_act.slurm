#!/bin/bash
#SBATCH --job-name=train_act
#SBATCH --account=kmarino
#SBATCH --partition=kmarino-gpu-grn
#SBATCH --qos=kmarino-gpu-grn
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1470293/projects/pusht/logs/%x-%j.out

# Train ACT on PushT, set up to compare against train_dp.slurm.
# Usage: sbatch train_act.slurm <run_name> <total_steps> <save_and_eval_every> <eval_episodes>
#   smoke test: sbatch train_act.slurm act_smoke 1000 500 10
#   full run:   sbatch train_act.slurm act_run1 200000 25000 50
# If the run already has a checkpoint, the job resumes from it (rerun the same command after a timeout).
#
# There is no official ACT PushT checkpoint. LeRobot's ACT defaults target ALOHA (50 Hz): chunk 100 = 2 s.
# PushT runs at 10 Hz, so chunk 100 would be 10 s open loop. We match DP's timing instead
# (predict 16, execute 8) and keep the rest of ACT's defaults, including the ImageNet ResNet18 backbone
# (pre-downloaded to $TORCH_HOME). Data, batch size, steps, seed and eval settings match train_dp.slurm.

RUN=${1:?run name}
STEPS=${2:-200000}
FREQ=${3:-25000}
EVAL_EPS=${4:-50}

module load miniforge3/25.11.0
eval "$(conda shell.bash hook)"
conda activate ~/envs/lerobot

export HF_HOME=/scratch/general/vast/u1470293/pusht/hf_cache
export TORCH_HOME=$HF_HOME/torch     # ImageNet ResNet18 weights, downloaded on the login node
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export SDL_VIDEODRIVER=dummy
SCRATCH=/scratch/general/vast/u1470293/pusht
OUT=$SCRATCH/outputs/train/$RUN
LAST=$OUT/checkpoints/last/pretrained_model/train_config.json

if [ -f "$LAST" ]; then
  echo "Resuming $RUN from $(readlink -f $OUT/checkpoints/last)"
  RESUME=(--resume=true --config_path=$LAST)
else
  echo "Starting new run $RUN"
  RESUME=()
fi
echo "Job $SLURM_JOB_ID on $(hostname) at $(date): steps=$STEPS, save/eval every $FREQ, $EVAL_EPS eval episodes"

lerobot-train \
  "${RESUME[@]}" \
  --dataset.repo_id=lerobot/pusht \
  --dataset.root=$SCRATCH/data/lerobot/pusht \
  --policy.type=act \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --env.type=pusht \
  --seed=100000 \
  --batch_size=64 \
  --num_workers=4 \
  --steps=$STEPS \
  --save_freq=$FREQ \
  --env_eval_freq=$FREQ \
  --log_freq=100 \
  --eval.n_episodes=$EVAL_EPS \
  --eval.batch_size=$EVAL_EPS \
  --eval.use_async_envs=false \
  --output_dir=$OUT \
  --job_name=$RUN
STATUS=$?

echo "Exit code $STATUS at $(date)"
exit $STATUS    # pass lerobot-train's result to Slurm, so a crash shows as FAILED, not COMPLETED
