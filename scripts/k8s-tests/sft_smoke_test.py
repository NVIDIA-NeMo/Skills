#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real SFT smoke test: fine-tune GPT-2 (124M) on synthetic math Q&A data.

Submits via Pipeline+KubernetesBackend for Track 1 validation.

Usage:
    # Single-node 2-GPU SFT
    .venv/bin/python scripts/k8s-tests/sft_smoke_test.py --gpus 2

    # Multi-node 2x2 SFT
    .venv/bin/python scripts/k8s-tests/sft_smoke_test.py --gpus 2 --nodes 2
"""

import argparse
import os
import sys
from pathlib import Path

# Bootstrap imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nemo_run as run  # noqa: E402

from nemo_skills.pipeline.utils.declarative import (  # noqa: E402
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)

# The SFT training script that runs inside the container.
# Fine-tunes GPT-2 (124M params) on 100 synthetic math Q&A examples.
SFT_TRAIN_SCRIPT = r"""
import json, os, sys, tempfile

# Workaround: NGC 26.01 ships apex without amp, which breaks transformers' import.
# Patch it so transformers skips the failing import path gracefully.
try:
    import apex
    if not hasattr(apex, "amp"):
        import types
        apex.amp = types.ModuleType("apex.amp")
        sys.modules["apex.amp"] = apex.amp
except ImportError:
    pass

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import Dataset

# 1. Create synthetic SFT dataset (100 math Q&A examples)
data = []
for i in range(100):
    data.append({"input": f"What is {i} + {i}?", "output": f"The answer is {i+i}."})

print(f"Created {len(data)} synthetic SFT examples")

# 2. Load GPT-2 tokenizer and model
model_name = "gpt2"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# 3. Tokenize dataset
class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.examples = []
        for item in data:
            text = f"Q: {item['input']}\nA: {item['output']}"
            encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            encoded["labels"] = encoded["input_ids"].copy()
            self.examples.append({k: torch.tensor(v) for k, v in encoded.items()})
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

dataset = SFTDataset(data, tokenizer)
print(f"Dataset tokenized: {len(dataset)} examples")

# 4. Training
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

output_dir = tempfile.mkdtemp(prefix="sft_smoke_")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    bf16=torch.cuda.is_bf16_supported(),
    dataloader_num_workers=0,
    ddp_backend="nccl",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"[Rank {rank}/{world_size}] Starting SFT training on GPU {local_rank}")
trainer.train()

if rank == 0:
    final_loss = trainer.state.log_history[-1].get("train_loss", "N/A")
    print(f"\n=== SFT SMOKE TEST PASSED ===")
    print(f"  Model: {model_name} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    print(f"  Dataset: 100 synthetic math Q&A examples")
    print(f"  Epochs: 2")
    print(f"  World size: {world_size}")
    print(f"  Final loss: {final_loss}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
"""


def main():
    parser = argparse.ArgumentParser(description="SFT smoke test via Pipeline+KubernetesBackend")
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--image", default=os.environ.get("PYTORCH_IMAGE", "nvcr.io/nvidia/pytorch:25.03-py3"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mode = "multi" if args.nodes > 1 else "single"
    print(f"SFT Smoke Test | Mode: {mode} | Nodes: {args.nodes} | GPUs/node: {args.gpus}")
    print(f"Image: {args.image} | Namespace: {args.namespace}")

    # Build the torchrun command
    if args.nodes > 1:
        cmd = (
            "export NCCL_DEBUG=INFO && export NCCL_DEBUG_SUBSYS=INIT,NET && "
            f"cat > /tmp/sft_train.py << 'PYEOF'\n{SFT_TRAIN_SCRIPT}\nPYEOF\n"
            f"torchrun --nproc_per_node={args.gpus} --nnodes={args.nodes} "
            f"--node_rank=${{NODE_RANK:-0}} --master_addr=${{MASTER_ADDR:-localhost}} "
            f"--master_port=${{MASTER_PORT:-29500}} /tmp/sft_train.py"
        )
    else:
        cmd = (
            "export NCCL_DEBUG=INFO && export NCCL_DEBUG_SUBSYS=INIT,NET && "
            f"cat > /tmp/sft_train.py << 'PYEOF'\n{SFT_TRAIN_SCRIPT}\nPYEOF\n"
            f"torchrun --nproc_per_node={args.gpus} --master_port=29500 /tmp/sft_train.py"
        )

    script = run.Script(inline=cmd)
    command = Command(script=script, container="nemo-skills", name="sft-trainer")
    group = CommandGroup(
        commands=[command],
        hardware=HardwareConfig(num_gpus=args.gpus, num_nodes=args.nodes),
        name="sft-smoke",
        log_dir="/tmp/sft-smoke-logs",
    )

    cluster_config = {
        "executor": "kubernetes",
        "namespace": args.namespace,
        "containers": {"nemo-skills": args.image},
        "skip_hf_home_check": True,
        "default_timeout": "30m",
        "env_vars": ["NCCL_DEBUG=INFO"],
    }

    pipeline = Pipeline(
        name="sft-smoke-test",
        cluster_config=cluster_config,
        jobs=[{"name": "sft-smoke", "group": group}],
    )

    if args.dry_run:
        pipeline.run(dry_run=True)
        print("Dry run complete.")
        return

    result = pipeline.run(dry_run=False)
    print(f"\nSubmitted {len(result)} job(s)")
    for name, handle in result.items():
        print(f"  {name}: job_id={handle.job_id}")

    # Wait and collect logs
    from nemo_skills.pipeline.backends import JobStatus, get_backend

    backend = get_backend(cluster_config)

    for name, handle in result.items():
        print(f"\nWaiting for job '{name}'...")
        status = backend.wait_for_completion(handle, timeout=1800)
        print(f"Status: {status.value}")
        if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            print("\n--- Logs ---")
            for line in backend.get_logs(handle):
                print(line, end="" if line.endswith("\n") else "\n")
        if status == JobStatus.FAILED:
            sys.exit(1)

    print("\n=== ALL JOBS COMPLETED ===")


if __name__ == "__main__":
    main()
