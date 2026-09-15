#!/bin/bash
#SBATCH --job-name="EaglePerceiverResampler eval"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=80G
#SBATCH --mail-user=mnov0010@student.monash.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Evaluates a trained EaglePerceiverResampler checkpoint: MT-bench + GSM8K
# speculative generation vs. autoregressive baseline, then the speedup ratio.
#
# Submit:  sbatch --export=CKPT=$DIR/runs/perceiver-llama3-8b/state_0 m3_scripts/eval-perceiver-llama3-8b.sbatch
# (CKPT must be a directory written by save_16bit_model, e.g. .../state_0)

set -euo pipefail

nvidia-smi

module load cuda/12.8

DIR=/home/michaln/ml20_scratch/michaln/EAGLE
eval "$("$DIR"/bin/micromamba shell hook -s bash -r ~/micromamba)"
micromamba activate "$DIR"/venv/eagle-py312

pip3 install -r requirements.txt
pip3 install deepspeed datasets

BASE="$DIR/models/Llama-3.1-8B-Instruct"
: "${CKPT:?set CKPT to a checkpoint dir, e.g. via sbatch --export=CKPT=.../state_0}"

# Export an inference-ready checkpoint (backfills d2t/t2d + embed_tokens,
# copies the perceiver config as config.json).
EXPORT="$DIR/evals/ckpt-$(basename "$CKPT")"
if [[ ! -f "$EXPORT/pytorch_model.bin" ]]; then
    python3 "$DIR/m3_scripts/export_checkpoint.py" \
        --ckpt "$CKPT" \
        --config "$DIR/eagle/traineagle3/config_perceiver_llama3_8b.json" \
        --base "$BASE" \
        --cache "$DIR/eagle/traineagle3/cache.pt" \
        --out "$EXPORT"
fi

cd "$DIR"
mkdir -p "$DIR/evals"

for BENCH in mt_bench gsm8k; do
    EA_FILE="$DIR/evals/$BENCH/perceiver-llama3-8b.jsonl"
    BASE_FILE="$DIR/evals/$BENCH/baseline-llama3-8b.jsonl"

    python3 -m eagle.evaluation.gen_ea_answer_llama3chat \
        --ea-model-path "$EXPORT" \
        --base-model-path "$BASE" \
        --use_eagle3 \
        --model-id perceiver-llama3-8b \
        --bench-name "$BENCH" \
        --answer-file "$EA_FILE" \
        --max-new-token 1024 \
        --num-gpus-total 1

    if [[ ! -f "$BASE_FILE" ]]; then
        python3 -m eagle.evaluation.gen_baseline_answer_llama3chat \
            --base-model-path "$BASE" \
            --ea-model-path "$EA_PATH" \
            --model-id baseline-llama3-8b \
            --bench-name "$BENCH" \
            --answer-file "$BASE_FILE" \
            --max-new-token 1024 \
            --num-gpus-total 1
    fi

    echo "=== speedup ratio ($BENCH) ==="
    python3 eagle/evaluation/speed.py \
        --ea-file "$EA_FILE" \
        --base-file "$BASE_FILE" \
        --tokenizer "$BASE"
done
