"""Trainable prefix vectors and periodic-injection logic.

This module contains the two core classes for SSM prefix tuning:

PrefixEncoder
    Holds K trainable "soft" prefix vectors in the model's embedding space.
    These are the only parameters updated during prefix tuning; all backbone
    weights remain frozen.

PeriodicPrefixInjector
    Addresses the fading-memory limitation of SSMs by re-inserting the prefix
    at fixed intervals throughout the token sequence.

The Fading-Memory Problem

In a Mamba SSM, the hidden state evolves as:

    h_t = A · h_{t-1} + B · x_t
    y_t = C · h_t

where A is a diagonal matrix with entries in (0, 1) (discrete-time eigenvalues
after the zero-order-hold discretisation of Δ).  The prefix's contribution to
the hidden state at position t is:

    h_t^{prefix} = (∏_{i=1}^{t} A_i) · h_0^{prefix}

Because each A_i has entries strictly less than 1, this product decays
exponentially in t.  For SST-2 sentences with max_length = 128 the decay is
modest but measurable; for longer sequences it can be severe.

PeriodicPrefixInjector mitigates this by re-injecting the prefix every
`period` positions.  The worst-case distance from any real token to the
nearest prefix injection is then `period − 1`, so the maximum attenuation
factor is A^{period} rather than A^{T}.  This is especially important when
`period ≪ T`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ssm_prefix_tuning.config import PeriodicInjectionConfig, PrefixConfig


class PrefixEncoder(nn.Module):
    """K trainable prefix vectors in the model's hidden-state space.

    Two parameterisation modes (controlled by PrefixConfig.projection):

    Direct mode (default):
        self.embedding  = nn.Embedding(K, hidden_size)
        Parameters: K × hidden_size.

    Projected mode (Li & Liang, 2021 — "Prefix-Tuning"):
        self.embedding  = nn.Embedding(K, proj_hidden)
        self.transform  = Linear(proj_hidden, hidden_size) with Tanh activation.
        Total parameters: K × proj_hidden + proj_hidden × hidden_size.
        The bottleneck keeps the free parameter count small while the MLP
        smooths the optimisation landscape.
    """

    def __init__(self, config: PrefixConfig) -> None:
        super().__init__()
        self.prefix_length = config.prefix_length
        self.hidden_size = config.hidden_size
        self.use_projection = config.projection

        if config.projection:
            self.embedding = nn.Embedding(config.prefix_length, config.projection_hidden_size)
            self.transform = nn.Sequential(
                nn.Linear(config.projection_hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            self.embedding = nn.Embedding(config.prefix_length, config.hidden_size)
            self.transform = None

        # Initialise with a small normal distribution — the same scheme used
        # by HuggingFace for embedding tables.
        nn.init.normal_(self.embedding.weight, std=config.init_std)

    def forward(self) -> torch.Tensor:
        """Return prefix embeddings of shape [K, hidden_size].

        No batch dimension is included; callers expand to [B, K, D] as needed.
        """
        indices = torch.arange(self.prefix_length, device=self.embedding.weight.device)
        embeds = self.embedding(indices)          # [K, proj_hidden] or [K, D]
        if self.transform is not None:
            embeds = self.transform(embeds)       # [K, D]
        return embeds


class PeriodicPrefixInjector(nn.Module):
    """Re-inserts prefix embeddings at fixed intervals in the token sequence.

    Given token embeddings of shape [B, T, D] and a PrefixEncoder that
    produces prefix embeddings of shape [K, D], this module interleaves
    copies of the prefix into the sequence:

        output = [prefix | tok_0…tok_{N-1} | prefix | tok_N…tok_{2N-1} | …]

    The resulting sequence has length  T + K × ceil(T / period).

    An extended attention mask is also returned, with 1s at all injected
    prefix positions and the original mask values preserved for real tokens.

    Note: the gradients flow through prefix_encoder, so the prefix parameters
    continue to receive updates even inside this injector.
    """

    def __init__(
        self,
        config: PeriodicInjectionConfig,
        prefix_encoder: PrefixEncoder,
    ) -> None:
        super().__init__()
        self.period = config.period
        # Store as attribute (not sub-module) because PrefixEncoder is already
        # registered on the parent MambaPrefixModel.  We hold a reference only
        # so this module can call prefix_encoder() during forward.
        self.prefix_encoder = prefix_encoder

    def forward(
        self,
        token_embeds: torch.Tensor,   # [B, T, D]
        attention_mask: torch.Tensor, # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interleave prefix blocks into the token embedding sequence.

        Returns:
            augmented_embeds: [B, T + K × n_injections, D]
            augmented_mask:   [B, T + K × n_injections]  (long)
        """
        B, T, D = token_embeds.shape
        device = token_embeds.device

        prefix_embeds = self.prefix_encoder()                       # [K, D]
        K = prefix_embeds.size(0)
        # Expand prefix to batch size.
        prefix_block = prefix_embeds.unsqueeze(0).expand(B, -1, -1).to(device)  # [B, K, D]
        prefix_mask = torch.ones(B, K, dtype=torch.long, device=device)

        injection_indices = self._build_injection_indices(T)

        chunks_embeds: list[torch.Tensor] = []
        chunks_mask: list[torch.Tensor] = []
        prev = 0
        for idx in injection_indices:
            # Inject prefix before real tokens at position `idx`.
            chunks_embeds.append(prefix_block)
            chunks_mask.append(prefix_mask)
            if idx > prev:
                chunks_embeds.append(token_embeds[:, prev:idx, :])
                chunks_mask.append(attention_mask[:, prev:idx])
            prev = idx

        # Append any remaining real tokens after the last injection.
        if prev < T:
            chunks_embeds.append(token_embeds[:, prev:, :])
            chunks_mask.append(attention_mask[:, prev:])

        augmented_embeds = torch.cat(chunks_embeds, dim=1)   # [B, T', D]
        augmented_mask = torch.cat(chunks_mask, dim=1)       # [B, T']
        return augmented_embeds, augmented_mask

    def _build_injection_indices(self, T: int) -> list[int]:
        """Return the real-token positions before which a prefix block is inserted.

        Injection happens at indices 0, period, 2×period, … (i.e. at the start
        of every chunk of `period` real tokens).
        """
        return list(range(0, T, self.period))
