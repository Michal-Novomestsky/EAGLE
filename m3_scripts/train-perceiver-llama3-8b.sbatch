#!/bin/bash
#SBATCH --job-name="EaglePerceiverResampler LLaMA-3.1-8B"
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=200G
#SBATCH --mail-user=mnov0010@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Trains EaglePerceiverResampler (perceiver resampler over the entire target
# residual stream) against LLaMA-3.1-8B-Instruct on ShareGPT.
#
# Prereqs: bash m3_scripts/install-python.sh && bash m3_scripts/download-assets.sh
#          plus a one-time `wandb login` (key is persisted in ~/.netrc)
# Submit:  sbatch m3_scripts/train-perceiver-llama3-8b.sbatch
#
# NUM_GPUS is not read from the environment by SLURM; to change GPU count edit
# both --gres above and --num_gpus below.

nvidia-smi

module load cuda/12.8

DIR=/home/michaln/ml20_scratch/michaln/EAGLE
eval "$("$DIR"/bin/micromamba shell hook -s bash -r ~/micromamba)"
micromamba activate "$DIR"/venv/eagle-py312

export WANDB_PROJECT=${WANDB_PROJECT:-eagle-perceiver-resampler}

cd "$DIR/eagle/traineagle3"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True deepspeed --num_gpus 1 main.py \
    --deepspeed_config ds_config.json \
    --configpath config_perceiver_llama3_8b.json \
    --basepath "$DIR/models/Llama-3.1-8B-Instruct" \
    --trainpath "$DIR/data/sharegpt_train.jsonl" \
    --testpath "$DIR/data/sharegpt_test.jsonl" \
    --savedir "$DIR/runs/perceiver-llama3-8b_0-drop_pre-norm_2-layer"
