#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Lightweight GPT-2 SFT workload for real K8s pipeline validation.

"""Standalone GPT-2 mini-training workload used by Kubernetes SFT smoke tests."""

import os
import tempfile

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


class SFTDataset(Dataset):
    """Synthetic SFT dataset for smoke validation."""

    def __init__(self, tokenizer, n: int = 50):
        """Build ``n`` synthetic math QA examples and tokenize them."""
        self.examples = []
        for i in range(n):
            text = f"Q: What is {i} + {i}? A: The answer is {i + i}."
            enc = tokenizer(text, truncation=True, max_length=64, padding="max_length")
            enc["labels"] = enc["input_ids"].copy()
            self.examples.append({k: torch.tensor(v) for k, v in enc.items()})

    def __len__(self):
        """Return dataset size."""
        return len(self.examples)

    def __getitem__(self, i):
        """Return one tokenized training example by index."""
        return self.examples[i]


def main():
    """Run a short Trainer-based GPT-2 SFT step and print pass signal on rank 0."""
    rank = int(os.environ.get("RANK", 0))
    print(f"[Rank {rank}] Loading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print(f"[Rank {rank}] GPT-2 loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    dataset = SFTDataset(tokenizer)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=tempfile.mkdtemp(),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            logging_steps=5,
            save_strategy="no",
            report_to="none",
            bf16=torch.cuda.is_bf16_supported(),
            ddp_backend="nccl",
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    if rank == 0:
        loss = trainer.state.log_history[-1].get("train_loss", "N/A")
        print(f"PIPELINE RUN PASSED: model=gpt2 loss={loss}")


if __name__ == "__main__":
    main()
