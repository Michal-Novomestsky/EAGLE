#!/bin/bash
#SBATCH --job-name="EAGLE-3 baseline train LLaMA-3.1-8B"
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=200G
#SBATCH --mail-user=mnov0010@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Trains the VANILLA EAGLE-3 draft (concat-3-layers + FC, stock config.json) on
# the same ShareGPT data as the perceiver variant, for a like-for-like
# data/compute comparison. Usually unnecessary: the official pretrained draft
# (yuhuili/EAGLE3-LLaMA3.1-Instruct-8B) is downloaded by download-assets.sh and
# can be evaluated directly with eval-eagle3-llama3-8b.sbatch.
#
# Prereqs: bash m3_scripts/install-python.sh && bash m3_scripts/download-assets.sh
# Submit:  sbatch --export=WANDB_API_KEY=... m3_scripts/train-eagle3-llama3-8b.sbatch
#
# NOTE: run this in a different working copy (or before/after the perceiver run):
# main.py's scandata() writes cache.pt into this directory, which is shared.
#
# NUM_GPUS is not read from the environment by SLURM; to change GPU count edit
# both --gres above and --num_gpus below.

nvidia-smi

module load cuda/12.8

DIR=/home/michaln/ml20_scratch/michaln/EAGLE
eval "$("$DIR"/bin/micromamba shell hook -s bash -r ~/micromamba)"
micromamba activate "$DIR"/venv/eagle-py312

pip3 install -r requirements.txt
pip3 install deepspeed datasets

export WANDB_PROJECT=${WANDB_PROJECT:-eagle3-baseline}

cd "$DIR/eagle/traineagle3"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --num_gpus 1 main.py \
    --deepspeed_config ds_config.json \
    --configpath config.json \
    --basepath "$DIR/models/Llama-3.1-8B-Instruct" \
    --trainpath "$DIR/data/sharegpt_train.jsonl" \
    --testpath "$DIR/data/sharegpt_test.jsonl" \
    --savedir "$DIR/runs/eagle3-llama3-8b_last-layer"
