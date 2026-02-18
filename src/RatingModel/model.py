import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int((2 / 3) * (4 * config.embed_dim))

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
        self.self_attn = nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(config.embed_dim)
        self.cross_attn = nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.swiglu = SwiGLU(config)
    
    def forward(self, x, context):
        y = self.norm1(x)
        attn_out, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + attn_out
        y = self.norm_cross(x)
        cross_out, _ = self.cross_attn(query=y, key=context, value=context, need_weights=False)
        x = x + cross_out        
        x = x + self.swiglu(self.norm2(x))
        return x

class RatingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.FEN_embedding = nn.Embedding(config.n_fen_tokens, config.embed_dim)
        self.theme_embedding = nn.Linear(config.n_themes, config.embed_dim, bias=False)
        self.positional_embedding = nn.Embedding(config.fen_length, config.embed_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.regressor_head = nn.Linear(config.embed_dim, 1, bias=False)
        self.config = config

    def forward(self, fen_tokens, theme_tokens):
        pos = torch.arange(0, self.config.fen_length, device=fen_tokens.device)
        x = self.FEN_embedding(fen_tokens) + self.positional_embedding(pos)
        context = self.theme_embedding(theme_tokens).unsqueeze(1)
        for block in self.blocks:
            x = block(x, context)

        x = torch.mean(x, dim=1)
        # x = torch.flatten(x, start_dim=1)
        return self.regressor_head(x).squeeze(1)  # (batch_size,)
