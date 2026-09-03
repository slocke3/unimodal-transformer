"""Continuous-input transformer: a sinusoidal (Fourier) embedding of the raw
value replaces the learned per-bin lookup, giving the input a metric (nearby
values -> nearby vectors). The output stays a softmax over n_out bins, so this
is the "continuous in, discrete out" rung of the resolution ladder.
"""
import math

import torch
import torch.nn as nn

from .model import LearnedPositionalEmbedding


class SinusoidalValueEmbedding(nn.Module):
    """Map a continuous value x in [0, 1] to a d_model vector.

    A fixed geometric ladder of frequencies lifts the scalar into a
    multi-frequency sinusoidal basis (so high-frequency functions of x are
    linearly accessible, avoiding the spectral bias of a raw-scalar input);
    a learned linear projection then mixes it into the model width. The
    max_freq sets the finest x-scale the input can resolve -- an "effective
    input resolution" analogous to bin width.
    """
    def __init__(self, d_model, n_freqs=64, max_freq=256.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), n_freqs))
        self.register_buffer("freqs", freqs)               # (n_freqs,)
        self.proj = nn.Linear(2 * n_freqs, d_model)

    def forward(self, x):
        # x: (batch, seq_len) float in [0, 1]
        ang = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs  # (b, s, n_freqs)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(feats)                             # (b, s, d_model)


class ContinuousInputTransformer(nn.Module):
    """Same trunk as DiscreteTrajectoryTransformer, but the input is a raw
    value (embedded sinusoidally) and the output is a softmax over n_out bins.
    """
    def __init__(self, n_out=64, context_len=50, d_model=128, n_heads=4,
                 n_layers=4, d_ff=None, dropout=0.1, n_freqs=64, max_freq=256.0):
        super().__init__()
        self.n_out = n_out
        self.context_len = context_len
        self.d_model = d_model
        if d_ff is None:
            d_ff = 4 * d_model

        self.value_embed = SinusoidalValueEmbedding(d_model, n_freqs, max_freq)
        self.pos_embedding = LearnedPositionalEmbedding(context_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, n_out)

        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x):
        # x: (batch, seq_len) float in [0, 1]
        seq_len = x.size(1)
        h = self.value_embed(x)
        h = self.pos_embedding(h)
        h = self.transformer(h, mask=self.causal_mask[:seq_len, :seq_len], is_causal=True)
        return self.output_head(h[:, -1, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
