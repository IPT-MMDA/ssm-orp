"""SST-2 data loading and preprocessing.

SST-2 is a binary sentiment classification task (positive / negative movie
reviews) from the GLUE benchmark.  It is used here as the downstream
classification task for comparing prefix tuning, LoRA, and full fine-tuning.

Key preprocessing choice — left-padding:
    Mamba is a causal model: each position only processes previous positions.
    When a batch contains sequences of different lengths, we pad on the LEFT
    so the last real token of every sequence stays at index −1.  This lets
    MambaForSequenceClassification pool the last hidden state at position −1
    regardless of padding, which is the correct classification token.
"""

from __future__ import annotations

from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# SST-2 validation split is used for evaluation because the test-set labels
# are withheld on the HuggingFace Hub.
_TRAIN_SPLIT = "train"
_VAL_SPLIT = "validation"


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load the tokenizer for a given model name and configure it for Mamba.

    - padding_side = "left"  →  last real token is always at position −1.
    - pad_token = eos_token  →  Mamba checkpoints have no dedicated pad token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_sst2(
    model_name: str,
    max_length: int = 128,
    batch_size: int = 16,
    num_workers: int = 0,
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """Load SST-2 and return (train_loader, val_loader).

    Args:
        model_name:         HuggingFace model name — used to load the matching tokenizer.
        max_length:         Sequences are padded / truncated to this length.
        batch_size:         Mini-batch size for both loaders.
        num_workers:        DataLoader worker count (0 = main process, safe on all OS).
        cache_dir:          Optional HuggingFace cache directory override.
        max_train_samples:  If set, subsample the training split to at most this many
                            examples.  Useful for quick CPU runs and memory-constrained
                            experiments.  The validation split is never subsampled.

    Returns:
        A tuple (train_loader, val_loader) of PyTorch DataLoaders.
    """
    tokenizer = get_tokenizer(model_name)
    raw = load_dataset("nyu-mll/glue", "sst2", cache_dir=cache_dir)

    def _tokenize(examples: dict) -> dict:
        return tokenizer(
            examples["sentence"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

    cols_to_remove = [c for c in raw["train"].column_names if c not in {"label"}]
    train_ds = raw[_TRAIN_SPLIT].map(_tokenize, batched=True, remove_columns=cols_to_remove)
    val_ds = raw[_VAL_SPLIT].map(_tokenize, batched=True, remove_columns=cols_to_remove)

    if max_train_samples is not None:
        train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    collator = Sst2Collator(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
    )
    return train_loader, val_loader


class Sst2Collator:
    """Collate function that standardises key names and stack tensors.

    The datasets library stores labels under the key 'label' (singular).
    HuggingFace models expect 'labels' (plural).  This collator renames
    the key and stacks all fields into batched tensors.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
