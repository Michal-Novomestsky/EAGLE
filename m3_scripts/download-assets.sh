#!/bin/bash
# Downloads the target model + ShareGPT training data for EaglePerceiverResampler.
# Requires hf auth login with access to meta-llama/Llama-3.1-8B-Instruct:
set -euo pipefail

DIR=/home/michaln/ml20_scratch/michaln/EAGLE
eval "$("$DIR"/bin/micromamba shell hook -s bash -r ~/micromamba)"
micromamba activate "$DIR"/venv/eagle-py312

mkdir -p "$DIR/models" "$DIR/data"

# --- Target model (override TARGET_MODEL to use a different target) ---
TARGET_MODEL=${TARGET_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TARGET_DIR="$DIR/models/$(basename "$TARGET_MODEL")"
hf download "$TARGET_MODEL" --local-dir "$TARGET_DIR"

# --- Official pretrained EAGLE-3 draft (public baseline; not gated) ---
EAGLE3_DIR="$DIR/models/EAGLE3-LLaMA3.1-Instruct-8B"
hf download yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --local-dir "$EAGLE3_DIR"

# --- ShareGPT training conversations (EAGLE-3 recipe) ---
SHAREGPT_JSON="$DIR/data/ShareGPT_V4.3_unfiltered_cleaned_split.json"
hf download Aeala/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir "$DIR/data"

# --- Split into train/test jsonl in the format traineagle3 expects ---
# (one {"id": ..., "conversations": [{"from": "human"|"gpt", "value": ...}, ...]} per line)
# Strip extra ShareGPT turn fields (text/markdown/...) so HF datasets can infer a
# uniform Arrow schema; training only reads "from" and "value".
python3 - <<'EOF'
import json, random, os
DIR = "/home/michaln/ml20_scratch/michaln/EAGLE"
src = os.path.join(DIR, "data", "ShareGPT_V4.3_unfiltered_cleaned_split.json")
train_out = os.path.join(DIR, "data", "sharegpt_train.jsonl")
test_out = os.path.join(DIR, "data", "sharegpt_test.jsonl")
with open(src) as f:
    data = json.load(f)
random.seed(0)
random.shuffle(data)
test, train = data[:100], data[100:]
def clean_convs(convs):
    out = []
    for t in convs or []:
        if not isinstance(t, dict):
            continue
        if "from" not in t or "value" not in t:
            continue
        out.append({"from": t["from"], "value": t["value"]})
    return out
for rows, path in ((train, train_out), (test, test_out)):
    n = 0
    with open(path, "w") as f:
        for row in rows:
            convs = clean_convs(row.get("conversations"))
            if not convs:
                continue
            f.write(json.dumps({"id": row["id"], "conversations": convs}) + "\n")
            n += 1
    print(f"wrote {n} -> {path}")
EOF