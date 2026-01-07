import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int((2 / 3) * (4 * config.embed_dim))  # Llama1 paper

        self.linear1 = nn.Linear(config.embed_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(config.embed_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, config.embed_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.linear1(x)
        swish = y * self.sigmoid(y)
        return self.linear3(swish * self.linear2(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.swiglu = SwiGLU(config)
    
    def forward(self, x):
        # check where normalization should be applied
        y = self.norm1(x)
        attn_output, _ = self.attention(y, y, y, need_weights=False)
        x = x + attn_output
        x = x + self.swiglu(self.norm2(x))
        return x

class MaskedDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.FEN_embedding = nn.Embedding(config.n_fen_tokens, config.embed_dim)
        self.theme_embedding = nn.Linear(config.n_themes, config.embed_dim, bias=False)  # look at how this should be added to the fen
        self.ratings_embedding = nn.Linear(config.rating_dim, config.embed_dim, bias=False)  # look at how this should be added to the fen

        self.positional_embedding = nn.Parameter(torch.zeros(1, config.fen_length, config.embed_dim))

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        self.classifier = nn.Linear(config.embed_dim, config.n_fen_tokens, bias=False)
        # self.classifier.weight = self.FEN_embedding.weight  # should we do this?

        self.config = config

    def forward(self, fen_tokens, theme_tokens, ratings):
        x = self.FEN_embedding(fen_tokens) + self.positional_embedding

        for block in self.blocks:
            x = block(x)

        logits = self.classifier(x)
        return logits
    
    def masking_schedule(self, t, schedule_type="linear", **kwargs):
        if schedule_type == "linear":
            return 1 - t
        else:
            raise RuntimeError()
        
    def masking_schedule_weight(self, t, schedule_type="linear", **kwargs):
        if schedule_type == "linear":
            return -1 / t
        else:
            raise RuntimeError()

    def elbo_loss(self, t, logits, true_fen_tokens, masked_fen_tokens):
        weight = self.masking_schedule_weight(t, schedule_type="linear")
        mask = masked_fen_tokens == self.config.mask_token
        loss = torch.sum(mask * weight * F.cross_entropy(logits.transpose(1, 2), true_fen_tokens, reduction="none"))
        return loss / len(logits)
