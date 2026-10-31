#!/bin/bash
#SBATCH --job-name=check_env
#SBATCH --account=kmarino
#SBATCH --partition=kmarino-gpu-grn
#SBATCH --qos=kmarino-gpu-grn
#SBATCH --gres=gpu:a800:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH -o /uufs/chpc.utah.edu/common/home/u1470293/projects/pusht/logs/%x-%j.out

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"

module load miniforge3/25.11.0
eval "$(conda shell.bash hook)"
conda activate ~/envs/lerobot

export HF_HOME=/scratch/general/vast/u1470293/pusht/hf_cache
export HF_HUB_OFFLINE=1          # no downloads inside jobs: everything must already be cached
export SDL_VIDEODRIVER=dummy     # pygame without a screen

python ~/projects/pusht/jobs/check_env.py

echo "Done at $(date)"
