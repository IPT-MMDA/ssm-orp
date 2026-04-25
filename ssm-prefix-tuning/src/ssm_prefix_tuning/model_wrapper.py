"""MambaPrefixModel — frozen Mamba backbone + trainable prefix.

Why MambaClassifier instead of MambaForSequenceClassification

transformers >= 5.x ships only MambaForCausalLM for the base Mamba architecture;
the MambaForSequenceClassification variant was never promoted to the main library.
MambaClassifier fills this gap: it wraps MambaModel with a single linear head
and exposes the same forward(input_ids, attention_mask, labels) interface that
both the training loop and the prefix-injection mechanism expect.

Model structure:
    MambaClassifier
    ├── backbone: MambaModel
    │   ├── embeddings: nn.Embedding(vocab_size, hidden_size)
    │   ├── layers: [MambaMixer × n_layers]
    │   └── norm_f: MambaRMSNorm
    └── classifier: nn.Linear(hidden_size, num_labels, bias=False)

Prefix injection strategy:
    1. Call backbone.embeddings(input_ids) to get token embeddings [B, T, D].
    2. Prepend K trainable prefix vectors  → [B, K+T, D].
    3. Optionally apply PeriodicPrefixInjector for fading-memory mitigation.
    4. Pass inputs_embeds directly to MambaClassifier, bypassing the internal
       embedding lookup in MambaModel.
    5. Pool the last hidden state at position -1 (always the last real token
       with left-padding — see data.py) and pass through the classifier head.

Freezing strategy (prefix tuning):
    - backbone.requires_grad_(False)      →  freezes all Mamba layers.
    - prefix_encoder.requires_grad_(True) →  only prefix vectors are trained.
    - classifier.requires_grad_(True)     →  classification head is always
      trained because SST-2 is a new task unseen by the pre-trained backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers import MambaConfig, MambaModel
from transformers.modeling_outputs import SequenceClassifierOutput

from ssm_prefix_tuning.config import (
    PeriodicInjectionConfig,
    PrefixConfig,
)
from ssm_prefix_tuning.prefix_encoder import PeriodicPrefixInjector, PrefixEncoder


class MambaClassifier(nn.Module):
    """MambaModel backbone with a linear classification head.

    This is a thin wrapper that adds sequence-classification capability to
    MambaModel, matching the interface of MambaForSequenceClassification that
    is absent from transformers >= 5.x.

    Pooling: the hidden state at the last sequence position (index −1) is used
    as the sentence representation.  When left-padding is applied (see data.py)
    the last position is always occupied by the final real token, making this
    pooling strategy correct regardless of sequence length.
    """

    def __init__(self, backbone: MambaModel, num_labels: int = 2) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_labels = num_labels
        self.classifier = nn.Linear(
            backbone.config.hidden_size, num_labels, bias=False
        )
        # Initialise the classification head with a small normal distribution.
        nn.init.normal_(self.classifier.weight, std=0.02)

    @classmethod
    def from_pretrained(cls, model_name: str, num_labels: int = 2) -> "MambaClassifier":
        """Load a pre-trained MambaModel and attach a fresh classification head."""
        backbone = MambaModel.from_pretrained(model_name)
        return cls(backbone, num_labels)

    @classmethod
    def from_config(cls, config: MambaConfig, num_labels: int = 2) -> "MambaClassifier":
        """Build from a MambaConfig without downloading weights (used in tests)."""
        backbone = MambaModel(config)
        return cls(backbone, num_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,  # absorbs PEFT/HF kwargs (output_attentions, output_hidden_states, …)
    ) -> SequenceClassifierOutput:
        """Run backbone + pool last token + classify.

        Args:
            input_ids:      Token ids [B, T].  Mutually exclusive with inputs_embeds.
            inputs_embeds:  Pre-computed embeddings [B, T, D].  When provided,
                            the backbone's embedding table is bypassed entirely.
            attention_mask: Padding mask [B, T] (1 = real token, 0 = pad).
                            Mamba's recurrent state does not use attention, so
                            this argument is accepted for interface compatibility
                            but has no effect on the SSM computation.
            labels:         Class indices [B] for computing cross-entropy loss.

        Returns:
            SequenceClassifierOutput with .loss (or None) and .logits [B, num_labels].
        """
        outputs = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        # Pool at the last position — with left-padding this is always the
        # final real token.  Shape: [B, D].
        pooled = outputs.last_hidden_state[:, -1, :]
        logits = self.classifier(pooled)  # [B, num_labels]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class MambaPrefixModel(nn.Module):
    """Frozen Mamba backbone with trainable prefix vectors for classification.

    Args:
        classifier_model: MambaClassifier whose backbone will be frozen.
        prefix_encoder:   PrefixEncoder holding K trainable prefix vectors.
        injector:         Optional PeriodicPrefixInjector — re-inserts the
                          prefix every `period` real-token positions to
                          mitigate the SSM fading-memory effect.
        num_labels:       Number of output classes (2 for SST-2).
    """

    def __init__(
        self,
        classifier_model: MambaClassifier,
        prefix_encoder: PrefixEncoder,
        injector: Optional[PeriodicPrefixInjector] = None,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.classifier_model = classifier_model
        self.prefix_encoder = prefix_encoder
        self.injector = injector
        self.num_labels = num_labels

        # Shortcut references used in forward.
        self._backbone = classifier_model.backbone
        self._embeddings = classifier_model.backbone.embeddings

        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze the Mamba backbone; keep prefix encoder and classifier trainable."""
        self._backbone.requires_grad_(False)
        self.prefix_encoder.requires_grad_(True)
        self.classifier_model.classifier.requires_grad_(True)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> SequenceClassifierOutput:
        """Run the prefix-augmented forward pass.

        Steps:
            1. Convert input_ids → token embeddings via backbone.embeddings.
            2. Prepend K trainable prefix vectors to every sequence.
            3. If periodic injection is enabled, also inject at every `period`
               real-token positions to mitigate the SSM fading-memory effect.
            4. Pass augmented embeddings to MambaClassifier via inputs_embeds,
               bypassing the internal embedding lookup.
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Step 1: token embeddings [B, T, D].
        token_embeds = self._embeddings(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)

        # Steps 2 & 3: build augmented sequence.
        if self.injector is not None:
            augmented_embeds, augmented_mask = self.injector(token_embeds, attention_mask)
        else:
            prefix_embeds = self.prefix_encoder()                           # [K, D]
            K = prefix_embeds.size(0)
            prefix_embeds = prefix_embeds.unsqueeze(0).expand(B, -1, -1)   # [B, K, D]
            prefix_mask = torch.ones(B, K, dtype=torch.long, device=device)
            augmented_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)  # [B, K+T, D]
            augmented_mask = torch.cat([prefix_mask, attention_mask], dim=1)    # [B, K+T]

        # Step 4: forward through classifier (backbone skips embedding lookup).
        return self.classifier_model(
            inputs_embeds=augmented_embeds,
            attention_mask=augmented_mask,
            labels=labels,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only the parameters that receive gradient updates."""
        return [p for p in self.parameters() if p.requires_grad]


def build_prefix_model(
    model_name: str,
    prefix_config: PrefixConfig,
    injection_config: PeriodicInjectionConfig,
    num_labels: int = 2,
) -> MambaPrefixModel:
    """Load a pre-trained Mamba model and wrap it with a trainable prefix."""
    classifier_model = MambaClassifier.from_pretrained(model_name, num_labels)
    prefix_config.hidden_size = classifier_model.backbone.config.hidden_size

    enc = PrefixEncoder(prefix_config)
    injector = (
        PeriodicPrefixInjector(injection_config, enc) if injection_config.enabled else None
    )
    return MambaPrefixModel(classifier_model, enc, injector, num_labels)


def build_prefix_model_from_config(
    hf_config: MambaConfig,
    prefix_config: PrefixConfig,
    injection_config: PeriodicInjectionConfig,
    num_labels: int = 2,
) -> MambaPrefixModel:
    """Build a MambaPrefixModel from a MambaConfig — used in tests without downloads."""
    classifier_model = MambaClassifier.from_config(hf_config, num_labels)
    prefix_config.hidden_size = hf_config.hidden_size

    enc = PrefixEncoder(prefix_config)
    injector = (
        PeriodicPrefixInjector(injection_config, enc) if injection_config.enabled else None
    )
    return MambaPrefixModel(classifier_model, enc, injector, num_labels)


def build_full_finetune_model(
    model_name: str,
    num_labels: int = 2,
) -> MambaClassifier:
    """Load a pre-trained Mamba model with all parameters unfrozen.

    All backbone weights and the classification head are updated during
    training.  Returns a MambaClassifier, which exposes the same
    forward(input_ids, attention_mask, labels) interface as MambaPrefixModel.
    """
    return MambaClassifier.from_pretrained(model_name, num_labels)
