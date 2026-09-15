"""Single-step gradient diagnostic: does autograd reach midlayer/perceiver?

Runs ONE forward/backward on a synthetic batch with plain loss.backward()
(no DeepSpeed), then prints per-parameter grad RMS for the draft modules.
Run from eagle/traineagle3 (needs cache.pt for t2d; scandata reuses it).

Usage:
    python grad_diagnostic.py --gradient_checkpoint 1 --configpath config_perceiver_llama3_8b.json
    python grad_diagnostic.py --gradient_checkpoint 0 --configpath config_perceiver_llama3_8b.json

Interpretation:
    - nonzero midlayer/perceiver grads here but zeros in the DeepSpeed
      grad_log.jsonl => ZeRO-2 param.grad harvesting artifact (training fine).
    - zero grads with checkpointing on, nonzero with it off => the
      torch.utils.checkpoint call is severing the graph.
    - zero in both => genuine graph break elsewhere.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--basepath", type=str, required=True)
parser.add_argument("--trainpath", type=str, required=True)
parser.add_argument("--configpath", type=str, default="config_perceiver_llama3_8b.json")
parser.add_argument("--gradient_checkpoint", type=int, default=1)
parser.add_argument("--seq_len", type=int, default=128)
args = parser.parse_args()

import torch
from configs import EConfig
from cnets import Model

train_config = {
    "bs": 1,
    "num_epochs": 1,
    "num_workers": 0,
    "max_len": 1024,
    "config_path": args.configpath,
    "gradient_checkpoint": bool(args.gradient_checkpoint),
}

config = EConfig.from_pretrained(args.configpath)
model = Model(config, None, train_config, path=args.basepath, load_emb=True, load_head=True)
model.scandata(args.trainpath, args.basepath)  # loads t2d from cache.pt if present
# DeepSpeed's bf16 mode casts the whole draft to bf16 during training; match
# that here, otherwise the bf16 target hidden states hit fp32 draft weights.
model = model.to(torch.bfloat16)
model.cuda().train()

torch.manual_seed(0)
B, S = 1, args.seq_len
input_ids = torch.randint(0, config.vocab_size, (B, S)).cuda()
attention_mask = torch.ones_like(input_ids)
loss_mask = torch.ones_like(input_ids)

plosses, _, acces = model(input_ids=input_ids, attention_mask=attention_mask, loss_mask=loss_mask)
loss = sum(0.8 ** i * p for i, p in enumerate(plosses))
loss.backward()

print(f"\n=== gradient_checkpoint={args.gradient_checkpoint} config={args.configpath} ===")
print(f"loss={loss.item():.4f} acces={[round(a, 4) for a in acces]}")

rows = []
for name, p in model.named_parameters():
    if name.startswith(("midlayer", "perceiver", "lm_head", "norm")):
        if p.grad is None:
            rows.append((name, None))
        else:
            rows.append((name, p.grad.float().pow(2).mean().sqrt().item()))

# Aggregate to keep output compact: exact per-param for perceiver, bucketed rest.
buckets = {}
for name, rms in rows:
    key = name.rsplit(".", 1)[0] if name.startswith("perceiver.") else name.split(".")[0]
    if rms is None:
        buckets.setdefault(key, []).append(None)
    else:
        buckets.setdefault(key, []).append(rms)

for key in sorted(buckets):
    vals = buckets[key]
    n_none = sum(v is None for v in vals)
    real = [v for v in vals if v is not None]
    rms = sum(v * v for v in real) / max(len(real), 1)
    print(f"{key:45s} grad RMS = {rms ** 0.5:.3e}   (params: {len(vals)}, grad=None: {n_none})")
