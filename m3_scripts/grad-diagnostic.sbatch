#!/bin/bash
#SBATCH --job-name="EAGLE grad diagnostic"
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=200G
#SBATCH --mail-user=mnov0010@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# One-step gradient diagnostic (no DeepSpeed): checks whether autograd reaches
# midlayer/perceiver, with gradient checkpointing on and off. See the docstring
# in eagle/traineagle3/grad_diagnostic.py for how to read the output.
#
# NOTE: run this in the same working copy as the perceiver training run so it
# reuses the existing cache.pt in eagle/traineagle3 (otherwise scandata
# rebuilds it, which is slow).
#
# Submit: sbatch m3_scripts/grad-diagnostic.sbatch

nvidia-smi

module load cuda/12.8

DIR=/home/michaln/ml20_scratch/michaln/EAGLE
eval "$("$DIR"/bin/micromamba shell hook -s bash -r ~/micromamba)"
micromamba activate "$DIR"/venv/eagle-py312

cd "$DIR/eagle/traineagle3"

for GC in 1 0; do
    echo "================ gradient_checkpoint=$GC ================"
    python grad_diagnostic.py \
        --gradient_checkpoint $GC \
        --configpath config_perceiver_llama3_8b.json \
        --basepath "$DIR/models/Llama-3.1-8B-Instruct" \
        --trainpath "$DIR/data/sharegpt_train.jsonl"
done
