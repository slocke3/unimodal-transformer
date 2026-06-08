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