import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.FEN_embedding = nn.Embedding(config.n_fen_tokens + 1, config.embed_dim)  # one additional mask token
        self.theme_embedding = nn.Linear(config.n_themes, config.embed_dim, bias=False)
        self.ratings_embedding = nn.Linear(config.rating_dim, config.embed_dim, bias=False)  # TODO: perhaps should use bias here
        self.positional_embedding = nn.Parameter(torch.zeros(1, config.fen_length, config.embed_dim))

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.classifier = nn.Linear(config.embed_dim, config.n_fen_tokens, bias=False)

        self.config = config

    def forward(self, fen_tokens, theme_tokens, ratings):
        x = self.positional_embedding + self.FEN_embedding(fen_tokens)
        x = x + self.theme_embedding(theme_tokens).unsqueeze(1) + self.ratings_embedding(ratings.unsqueeze(1)).unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        logits = self.classifier(x)
        return logits

    def elbo_loss(self, t, logits, true_fen_tokens, masked_fen_tokens):
        weight = self.config.masking_schedule.get_weight(torch.as_tensor(t)).unsqueeze(1)
        assert weight.ndim == 2
        mask = masked_fen_tokens == self.config.mask_token
        loss = torch.sum(mask * weight * F.cross_entropy(logits.transpose(1, 2), true_fen_tokens, reduction="none"))
        return loss / len(logits)
    
    # def log_prob(self, true_fen_tokens, theme_tokens, ratings):
    #     fen_tokens = torch.full_like(true_fen_tokens, self.config.mask_token)
    #     logits = self(fen_tokens, theme_tokens, ratings)
    #     log_probs = F.log_softmax(logits, dim=2)
    #     target_log_probs = torch.gather(log_probs, dim=2, index=true_fen_tokens.unsqueeze(-1))
    #     return torch.sum(target_log_probs, dim=1).squeeze(1)

    @torch.no_grad()
    def sample(self, theme_tokens, ratings, steps=256):
        batch_size = len(ratings)
        device = ratings.device
        fen_length = self.config.fen_length
        mask_token = self.config.mask_token
    
        fen = torch.full((batch_size, fen_length), mask_token, device=device, dtype=torch.long)  # start the fen so that all tokens are masked

        T_grid = torch.linspace(0, 1, steps + 1, device=device).to(device)
        for i in range(steps, 0, -1):
            t = T_grid[i]
            s = T_grid[i - 1]
            alpha_t = self.config.masking_schedule(t)
            alpha_s = self.config.masking_schedule(s)
            
            logits = self(fen, theme_tokens, ratings) 
            probs = F.softmax(logits, dim=2)
            
            p_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-9)
            p_mask = (1.0 - alpha_s) / (1.0 - alpha_t + 1e-9)
            probs = torch.cat([probs * p_unmask, torch.full((batch_size, fen_length, 1), p_mask, device=device, dtype=probs.dtype)], dim=2)

            is_masked = (fen == mask_token)
            if is_masked.any():
                flattened_probs = probs.view(-1, self.config.n_fen_tokens + 1)
                new_samples = torch.multinomial(flattened_probs, num_samples=1).view(batch_size, fen_length)
                fen = torch.where(is_masked, new_samples, fen)
            else:
                break
                
        return fen
