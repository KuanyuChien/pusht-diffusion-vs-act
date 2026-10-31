#!/bin/bash
#SBATCH --job-name=eval_official
#SBATCH --account=kmarino
#SBATCH --partition=kmarino-gpu-grn
#SBATCH --qos=kmarino-gpu-grn
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1470293/projects/pusht/logs/%x-%j.out

# Evaluate the official lerobot/diffusion_pusht (migrated to the new processor format) in gym-pusht.
# Usage: sbatch eval_official.slurm <n_episodes>     (default 10)

N=${1:-10}
BATCH=$(( N < 50 ? N : 50 ))     # episodes stepped together (one batched GPU call per step)
# use_async_envs=false: async workers start via forkserver and never import gym_pusht
# (NamespaceNotFound in LeRobot 0.6.2). The official eval also ran with use_async_envs=False.

echo "Job $SLURM_JOB_ID on $(hostname) at $(date): $N episodes, batch $BATCH"

module load miniforge3/25.11.0
eval "$(conda shell.bash hook)"
conda activate ~/envs/lerobot

export HF_HOME=/scratch/general/vast/u1470293/pusht/hf_cache
export HF_HUB_OFFLINE=1
export SDL_VIDEODRIVER=dummy
SCRATCH=/scratch/general/vast/u1470293/pusht

lerobot-eval \
  --policy.path=$SCRATCH/models/diffusion_pusht_migrated \
  --policy.device=cuda \
  --env.type=pusht \
  --eval.n_episodes=$N \
  --eval.batch_size=$BATCH \
  --eval.use_async_envs=false \
  --output_dir=$SCRATCH/outputs/eval/official_${N}ep_$SLURM_JOB_ID
STATUS=$?

echo "Exit code $STATUS at $(date)"
exit $STATUS    # pass lerobot-eval's result to Slurm, so a crash shows as FAILED, not COMPLETED
