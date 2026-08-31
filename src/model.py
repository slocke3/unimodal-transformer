import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0)


class DiscreteTrajectoryTransformer(nn.Module):
    """
    Causal transformer for next-token prediction on tokenized map trajectories.

    Input:  (batch, seq_len) integer bin indices
    Output: (batch, n_bins) logits for the next token
    """
    def __init__(self, n_bins=64, context_len=50, d_model=128,
                 n_heads=4, n_layers=4, d_ff=None, dropout=0.1):
        super().__init__()
        self.n_bins = n_bins
        self.context_len = context_len
        self.d_model = d_model

        if d_ff is None:
            d_ff = 4 * d_model

        self.token_embed = nn.Embedding(n_bins, d_model)
        self.pos_embedding = LearnedPositionalEmbedding(context_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, n_bins)

        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x):
        batch_size, seq_len = x.shape
        h = self.token_embed(x)
        h = self.pos_embedding(h)
        h = self.transformer(h, mask=self.causal_mask[:seq_len, :seq_len], is_causal=True)
        return self.output_head(h[:, -1, :])

    @torch.no_grad()
    def predict_rollout(self, x, n_steps, temperature=1.0):
        """Stochastic autoregressive rollout."""
        preds = []
        context = x.clone()
        for _ in range(n_steps):
            logits = self(context)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            preds.append(next_token)
            context = torch.cat([context[:, 1:], next_token], dim=1)
        return torch.cat(preds, dim=1)

    @torch.no_grad()
    def predict_rollout_greedy(self, x, n_steps):
        """Deterministic (argmax) autoregressive rollout."""
        preds = []
        context = x.clone()
        for _ in range(n_steps):
            logits = self(context)
            next_token = logits.argmax(dim=-1, keepdim=True)
            preds.append(next_token)
            context = torch.cat([context[:, 1:], next_token], dim=1)
        return torch.cat(preds, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DiscreteMLPBaseline(nn.Module):
    """MLP baseline: embed tokens, flatten, classify."""
    def __init__(self, n_bins=64, context_len=50, d_embed=32,
                 hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(n_bins, d_embed)
        flat_dim = context_len * d_embed
        layers = [nn.Linear(flat_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, n_bins))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.embed(x).flatten(1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ContinuousTrajectoryTransformer(nn.Module):
    """Causal transformer taking REAL x values instead of bin tokens.

    The token-embedding models see only which of n_bins the state fell in, so
    identifying the map means reconstructing a function from a coarsely
    quantised, unevenly sampled view of it. Feeding x directly removes that
    bottleneck: every context position is an exact (x_n, x_{n+1}) pair.

    Two output modes, so the input change can be separated from the output one:
      bins    logits over n_bins, cross-entropy against bin(x_{n+1}) -- the same
              output as the earlier runs, so only the INPUT differs.
      scalar  one real number, squared error against x_{n+1} -- continuous end
              to end.

    x is rescaled to [-1, 1] before the linear map. This is a pure
    reparameterisation -- Linear(1,d) on 2x-1 equals Linear(1,d) on x with
    weight 2w and bias b-w -- so it cannot change what the model can express,
    and norm_first=True puts a LayerNorm on the embedding immediately anyway.
    Kept as the conventional, better-conditioned starting point.
    """

    def __init__(self, n_bins=64, context_len=50, d_model=128, n_heads=4,
                 n_layers=4, d_ff=None, dropout=0.1, output_mode="bins"):
        super().__init__()
        assert output_mode in ("bins", "scalar")
        self.n_bins = n_bins
        self.context_len = context_len
        self.output_mode = output_mode
        if d_ff is None:
            d_ff = 4 * d_model

        self.input_proj = nn.Linear(1, d_model)
        self.pos_embedding = LearnedPositionalEmbedding(context_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, n_bins if output_mode == "bins" else 1)
        mask = torch.triu(torch.ones(context_len, context_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x, all_positions=False):
        """x: (batch, seq) float in [0,1]."""
        h = self.input_proj((2.0 * x - 1.0).unsqueeze(-1))
        h = self.pos_embedding(h)
        s = x.shape[1]
        h = self.transformer(h, mask=self.causal_mask[:s, :s], is_causal=True)
        out = self.output_head(h if all_positions else h[:, -1:, :])
        if self.output_mode == "scalar":
            out = out.squeeze(-1)
        return out if all_positions else (
            out[:, 0] if self.output_mode == "scalar" else out[:, 0, :])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
