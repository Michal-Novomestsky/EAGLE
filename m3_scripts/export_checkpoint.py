"""
Export a DeepSpeed-trained EAGLE draft checkpoint for inference.

DeepSpeed's save_16bit_model(exclude_frozen_parameters=True) drops frozen
parameters, so the saved state can be missing tensors that EaModel expects at
inference time (the draft<->target vocab mapping buffers d2t/t2d and the frozen
embedding matrix). This script produces a self-contained draft checkpoint:

    <out>/pytorch_model.bin   (weights + d2t/t2d + embed_tokens if missing)
    <out>/config.json         (copied from --config)

Usage:
    python3 m3_scripts/export_checkpoint.py \
        --ckpt runs/perceiver-llama3-8b/state_0 \
        --config eagle/traineagle3/config_perceiver_llama3_8b.json \
        --base models/Llama-3.1-8B-Instruct \
        --cache eagle/traineagle3/cache.pt \
        --out runs/perceiver-llama3-8b/export-state_0
"""

import argparse
import json
import os
import shutil

import torch


def load_state_dict(ckpt_dir):
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(bin_path):
        return torch.load(bin_path, map_location="cpu")
    if os.path.exists(st_path):
        from safetensors.torch import load_file

        return load_file(st_path)
    raise FileNotFoundError(
        f"no pytorch_model.bin or model.safetensors in {ckpt_dir}"
    )


def load_base_embedding(base_path):
    """Load the base model's embed_tokens.weight (same logic as cnets.py)."""
    from safetensors import safe_open

    try:
        with open(os.path.join(base_path, "model.safetensors.index.json"), "r") as f:
            index_json = json.loads(f.read())
        emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
        with safe_open(
            os.path.join(base_path, emb_path), framework="pt", device="cpu"
        ) as f:
            tensor_slice = f.get_slice("model.embed_tokens.weight")
            vocab_size, hidden_dim = tensor_slice.get_shape()
            return tensor_slice[:, :hidden_dim].float()
    except (FileNotFoundError, KeyError):
        with open(os.path.join(base_path, "pytorch_model.bin.index.json"), "r") as f:
            index_json = json.loads(f.read())
        emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
        weights = torch.load(os.path.join(base_path, emb_path), map_location="cpu")
        return weights["model.embed_tokens.weight"].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="checkpoint dir written by save_16bit_model (e.g. runs/.../state_0)")
    parser.add_argument("--config", required=True,
                        help="draft model config to copy as config.json (e.g. the perceiver config)")
    parser.add_argument("--base", required=True,
                        help="base model path, used to backfill embed_tokens if missing")
    parser.add_argument("--cache", default=None,
                        help="cache.pt written by Model.scandata during training (d2t/t2d)")
    parser.add_argument("--out", required=True, help="export directory")
    args = parser.parse_args()

    state_dict = load_state_dict(args.ckpt)
    print(f"loaded {len(state_dict)} tensors from {args.ckpt}")

    if "d2t" not in state_dict or "t2d" not in state_dict:
        assert args.cache is not None and os.path.exists(args.cache), \
            f"checkpoint is missing d2t/t2d and cache file not found: {args.cache}"
        cache = torch.load(args.cache, map_location="cpu")
        state_dict["d2t"] = cache["d2t"]
        state_dict["t2d"] = cache["t2d"]
        print(f"backfilled d2t/t2d from {args.cache}")

    if "embed_tokens.weight" not in state_dict:
        state_dict["embed_tokens.weight"] = load_base_embedding(args.base)
        print(f"backfilled embed_tokens.weight from {args.base}")

    os.makedirs(args.out, exist_ok=True)
    torch.save(state_dict, os.path.join(args.out, "pytorch_model.bin"))
    shutil.copy(args.config, os.path.join(args.out, "config.json"))
    print(f"exported inference-ready draft checkpoint to {args.out}")


if __name__ == "__main__":
    main()
