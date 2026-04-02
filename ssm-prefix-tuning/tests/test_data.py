"""Tests for data.py.

Unit tests use a tiny mock tokenizer / dataset so no network access is needed.
The slow integration test (marked with @pytest.mark.slow) downloads SST-2.
"""

import pytest
import torch

from ssm_prefix_tuning.data import Sst2Collator


# Helpers

class _FakeTokenizer:
    """Minimal tokenizer stub for unit tests."""
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"


def _make_batch(n: int, seq_len: int) -> list[dict]:
    """Create a fake batch of n tokenised examples of fixed length."""
    return [
        {
            "input_ids": torch.randint(1, 100, (seq_len,)),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "label": torch.tensor(i % 2),
        }
        for i in range(n)
    ]


# Collator tests

def test_collator_output_keys():
    """Collator returns exactly the three expected keys."""
    collator = Sst2Collator(_FakeTokenizer())
    batch = _make_batch(4, 10)
    out = collator(batch)
    assert set(out.keys()) == {"input_ids", "attention_mask", "labels"}


def test_collator_renames_label_to_labels():
    """The 'label' key from datasets is renamed to 'labels' for HF models."""
    collator = Sst2Collator(_FakeTokenizer())
    batch = _make_batch(4, 10)
    out = collator(batch)
    assert "labels" in out
    assert "label" not in out


def test_collator_batch_shapes():
    """Output tensors have the correct batch and sequence dimensions."""
    B, T = 4, 10
    collator = Sst2Collator(_FakeTokenizer())
    batch = _make_batch(B, T)
    out = collator(batch)
    assert out["input_ids"].shape == (B, T)
    assert out["attention_mask"].shape == (B, T)
    assert out["labels"].shape == (B,)


def test_collator_label_values():
    """Labels alternate 0/1 as constructed in _make_batch."""
    collator = Sst2Collator(_FakeTokenizer())
    batch = _make_batch(4, 8)
    out = collator(batch)
    expected = torch.tensor([0, 1, 0, 1])
    assert torch.equal(out["labels"], expected)


def test_collator_attention_mask_dtype():
    """Attention mask must be a long (int64) tensor for HF models."""
    collator = Sst2Collator(_FakeTokenizer())
    out = collator(_make_batch(2, 5))
    assert out["attention_mask"].dtype == torch.long


# Integration test — requires network access

@pytest.mark.slow
def test_load_sst2_integration():
    """SST-2 loads correctly; loaders yield batches with the right keys."""
    from ssm_prefix_tuning.data import load_sst2

    # Use a small dummy model name; we only need the tokenizer here.
    # state-spaces/mamba-130m-hf ships an EleutherAI/gpt-neox tokenizer.
    train_loader, val_loader = load_sst2(
        model_name="state-spaces/mamba-130m-hf",
        max_length=32,
        batch_size=8,
    )
    batch = next(iter(val_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[1] == 32
