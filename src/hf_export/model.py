import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ModelMixin, ConfigMixin

class SwiGLU(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        hidden_dim = int((2 / 3) * (4 * embed_dim))
        self.linear1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))

class Block(nn.Module):
    def __init__(self, embed_dim, n_heads, use_context=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.use_context = use_context
        if use_context:
            self.norm_cross = nn.LayerNorm(embed_dim)
            self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.swiglu = SwiGLU(embed_dim)
    
    def forward(self, x, context=None):
        y = self.norm1(x)
        attn_output, _ = self.attention(y, y, y, need_weights=False)
        x = x + attn_output
        if self.use_context and context is not None:
            y = self.norm_cross(x)
            cross_out, _ = self.cross_attn(query=y, key=context, value=context, need_weights=False)
            x = x + cross_out
        x = x + self.swiglu(self.norm2(x))
        return x

class MaskedDiffusion(ModelMixin, ConfigMixin):

    def __init__(
        self,
        n_fen_tokens=48,
        n_move_tokens=4,
        n_themes=66,
        rating_dim=1,
        fen_length=76,
        move_length=5,
        predict_moves=True,
        use_context=True,
        n_heads=8,
        n_layers=16,
        embed_dim=1024,
    ):
        super().__init__()
        # Register inputs to configuration dict for config.json compatibility
        self.register_to_config(
            n_fen_tokens=n_fen_tokens,
            n_move_tokens=n_move_tokens,
            n_themes=n_themes,
            rating_dim=rating_dim,
            fen_length=fen_length,
            move_length=move_length,
            predict_moves=predict_moves,
            use_context=use_context,
            n_heads=n_heads,
            n_layers=n_layers,
            embed_dim=embed_dim,
        )

        n_tokens = n_fen_tokens + (n_move_tokens if predict_moves else 0)
        self.mask_token = n_tokens
        self.seq_length = fen_length + (move_length if predict_moves else 0)

        self.FEN_embedding = nn.Embedding(n_tokens + 1, embed_dim)
        if use_context:
            self.theme_embedding = nn.Linear(n_themes, embed_dim, bias=False)
            self.ratings_embedding = nn.Linear(rating_dim, embed_dim, bias=True)
            
        self.positional_embedding = nn.Embedding(self.seq_length, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, n_heads, use_context) for _ in range(n_layers)])
        self.classifier = nn.Linear(embed_dim, n_tokens, bias=False)

    def forward(self, tokens, theme_tokens=None, ratings=None):
        pos = torch.arange(0, self.seq_length, dtype=torch.long, device=tokens.device)
        x = self.positional_embedding(pos) + self.FEN_embedding(tokens)

        if self.config.use_context:
            context = self.theme_embedding(theme_tokens).unsqueeze(1)
            emb_ratings = self.ratings_embedding(ratings.unsqueeze(1)).unsqueeze(1)
            context = torch.cat([context, emb_ratings], dim=1)
        else:
            context = None

        for block in self.blocks:
            x = block(x, context)

        logits = self.classifier(x)
        return logits
