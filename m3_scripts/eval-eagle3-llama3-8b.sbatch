#!/bin/bash
#SBATCH --job-name="EAGLE-3 baseline eval"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --mail-user=mnov0010@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Evaluates the vanilla EAGLE-3 baseline with the exact same pipeline/benches as
# m3_scripts/eval-perceiver-llama3-8b.sbatch, for a like-for-like comparison on
# M3 hardware. Public reference numbers (authors' hardware, EAGLE-3 NeurIPS 2025
# paper): MT-bench ~4.40x speedup (tau ~6.13), GSM8K ~4.48x (tau ~6.23).
#
# Mode 1 (default): official pretrained EAGLE-3 draft (yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)
#   sbatch m3_scripts/eval-eagle3-llama3-8b.sbatch
#
# Mode 2: self-trained vanilla EAGLE-3 checkpoint (from train-eagle3-llama3-8b.sbatch,
#         same ShareGPT data as the perceiver variant)
#   sbatch --export=CKPT=$DIR/runs/eagle3-llama3-8b/state_0 m3_scripts/eval-eagle3-llama3-8b.sbatch

set -euo pipefail

nvidia-smi

DIR=/home/michaln/ml20_scratch/michaln/EAGLE
eval "$("$DIR"/bin/micromamba shell hook -s bash -r ~/micromamba)"
micromamba activate "$DIR"/venv/eagle-py312

pip3 install -r requirements.txt
pip3 install deepspeed datasets

BASE="$DIR/models/Llama-3.1-8B-Instruct"

if [[ -n "${CKPT:-}" ]]; then
    # Self-trained checkpoint: export (backfills d2t/t2d + embed_tokens, copies config)
    EXPORT="$DIR/evals/ckpt-eagle3-$(basename "$CKPT")"
    if [[ ! -f "$EXPORT/pytorch_model.bin" ]]; then
        python3 "$DIR/m3_scripts/export_checkpoint.py" \
            --ckpt "$CKPT" \
            --config "$DIR/eagle/traineagle3/config.json" \
            --base "$BASE" \
            --cache "$DIR/eagle/traineagle3/cache.pt" \
            --out "$EXPORT"
    fi
    EA_PATH="$EXPORT"
    MODEL_ID=eagle3-llama3-8b-selftrained
else
    EA_PATH="$DIR/models/EAGLE3-LLaMA3.1-Instruct-8B"
    MODEL_ID=eagle3-llama3-8b
fi

cd "$DIR"
mkdir -p "$DIR/evals"

for BENCH in mt_bench gsm8k; do
    EA_FILE="$DIR/evals/$BENCH/$MODEL_ID.jsonl"
    BASE_FILE="$DIR/evals/$BENCH/baseline-llama3-8b.jsonl"

    python3 -m eagle.evaluation.gen_ea_answer_llama3chat \
        --ea-model-path "$EA_PATH" \
        --base-model-path "$BASE" \
        --use_eagle3 \
        --model-id "$MODEL_ID" \
        --bench-name "$BENCH" \
        --answer-file "$EA_FILE" \
        --max-new-token 1024 \
        --num-gpus-total 1

    # Shared with the perceiver eval: the baseline is only generated once.
    if [[ ! -f "$BASE_FILE" ]]; then
        python3 -m eagle.evaluation.gen_baseline_answer_llama3chat \
            --base-model-path "$BASE" \
            --model-id baseline-llama3-8b \
            --bench-name "$BENCH" \
            --answer-file "$BASE_FILE" \
            --max-new-token 1024 \
            --num-gpus-total 1
    fi

    echo "=== speedup ratio ($BENCH, $MODEL_ID) ==="
    python3 eagle/evaluation/speed.py \
        --ea-file "$EA_FILE" \
        --base-file "$BASE_FILE" \
        --tokenizer "$BASE"
done
