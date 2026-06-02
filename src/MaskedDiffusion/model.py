import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int((2 / 3) * (4 * config.embed_dim))

        self.linear1 = nn.Linear(config.embed_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(config.embed_dim, hidden_dim, bias=False)
        self.linear3 = nn.Linear(hidden_dim, config.embed_dim, bias=False)

    def forward(self, x):
        return self.linear3(F.silu(self.linear1(x)) * self.linear2(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)
        if config.use_context:
            self.norm_cross = nn.LayerNorm(config.embed_dim)
            self.cross_attn = nn.MultiheadAttention(config.embed_dim, config.n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.swiglu = SwiGLU(config)
    
    def forward(self, x, context=None):
        y = self.norm1(x)
        attn_output, _ = self.attention(y, y, y, need_weights=False)
        x = x + attn_output
        if context is not None:
            y = self.norm_cross(x)
            cross_out, _ = self.cross_attn(query=y, key=context, value=context, need_weights=False)
            x = x + cross_out
        x = x + self.swiglu(self.norm2(x))
        return x

class MaskedDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.FEN_embedding = nn.Embedding(config.n_tokens + 1, config.embed_dim)  # one additional mask token
        if config.use_context:
            self.theme_embedding = nn.Linear(config.n_themes, config.embed_dim, bias=False)
            self.ratings_embedding = nn.Linear(config.rating_dim, config.embed_dim, bias=True)
        self.seq_length = config.fen_length + (config.move_length if config.predict_moves else 0)
        self.positional_embedding = nn.Embedding(self.seq_length, config.embed_dim)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.classifier = nn.Linear(config.embed_dim, config.n_tokens, bias=False)

        self.config = config

    def forward(self, tokens, theme_tokens=None, ratings=None, checkpoint_activations=False):
        pos = torch.arange(0, self.seq_length, dtype=torch.long, device=tokens.device)
        x = self.positional_embedding(pos) + self.FEN_embedding(tokens)

        if self.config.use_context:
            context = self.theme_embedding(theme_tokens).unsqueeze(1)
            emb_ratings = self.ratings_embedding(ratings.unsqueeze(1)).unsqueeze(1)
            context = torch.cat([context, emb_ratings], dim=1)
        else:
            context = None

        for block in self.blocks:
            if checkpoint_activations and self.training:
                x = checkpoint(block, x, context, use_reentrant=False)
            else:
                x = block(x, context)

        logits = self.classifier(x)
        return logits

    def elbo_loss(self, t, logits, true_tokens, masked_tokens):
        weight = self.config.masking_schedule.get_weight(torch.as_tensor(t)).unsqueeze(1).to(logits.device)
        assert weight.ndim == 2
        mask = masked_tokens == self.config.mask_token
        loss = -torch.sum(mask * weight * F.cross_entropy(torch.movedim(logits, 2, 1), true_tokens, reduction="none"), dim=1)
        return loss
    
    @torch.compile
    @torch.no_grad()
    def sample(self, theme_tokens=None, ratings=None, batch_size=1, steps=256, temperature=1.0, generate_move_last=True, compute_kl=False, compute_entropy=False, ref_model=None):
        if theme_tokens is not None:
            batch_size = len(theme_tokens)
            device = theme_tokens.device
        else:
            device = next(self.parameters()).device
            
        mask_token = self.config.mask_token

        if compute_kl:
            assert ref_model is not None, "ref_model must be provided if compute_kl is True"
            total_kl_divergence = torch.zeros(batch_size, device=device)
        if compute_entropy:
            total_entropy = torch.zeros(batch_size, device=device)
        
        tokens = torch.full((batch_size, self.seq_length), mask_token, device=device, dtype=torch.long)
        T_grid = torch.linspace(0, 1, steps + 1, device=device).to(device)

        if not self.config.predict_moves:
            generate_move_last = False
        if generate_move_last:
            phases = [(0, self.config.fen_length), (self.config.fen_length, self.seq_length)]
        else:
            phases = [(0, self.seq_length)]

        for start_idx, end_idx in phases:
            for i in range(steps, 0, -1):
                t = T_grid[i]
                s = T_grid[i - 1]
                alpha_t = self.config.masking_schedule(t)
                alpha_s = self.config.masking_schedule(s)
                if s == 0.0:
                    alpha_s = torch.ones_like(alpha_s)
                
                logits = self(tokens, theme_tokens, ratings) 
                probs = F.softmax(logits / temperature, dim=2)
                
                p_unmask = (alpha_s - alpha_t) / (1.0 - alpha_t + 1e-13)
                p_mask = (1.0 - alpha_s) / (1.0 - alpha_t + 1e-13)
                
                probs = torch.cat([probs * p_unmask, torch.full((batch_size, self.seq_length, 1), p_mask, device=device, dtype=probs.dtype)], dim=2)

                # Gumbel-max sampling
                log_probs = torch.log(probs + 1e-13)
                u = torch.rand_like(log_probs)
                gumbel_noise = -torch.log(-torch.log(u + 1e-13) + 1e-13)
                new_samples = torch.argmax(log_probs + gumbel_noise, dim=-1)
                
                is_masked = (tokens == mask_token)
                in_window = torch.zeros_like(is_masked, dtype=torch.bool)
                in_window[:, start_idx:end_idx] = True
                is_updatable = is_masked & in_window

                if compute_kl or compute_entropy:
                    model_log_probs = F.log_softmax(logits / temperature, dim=2)
                    mask = is_updatable.float()
                    
                    if compute_kl:
                        ref_logits = ref_model(tokens, theme_tokens, ratings)
                        ref_log_probs = F.log_softmax(ref_logits / temperature, dim=2)
                        kl_div_vocab = F.kl_div(input=ref_log_probs, target=model_log_probs, log_target=True, reduction="none").sum(dim=2)
                        step_kl = (kl_div_vocab * mask * p_unmask).sum(dim=1)
                        total_kl_divergence = total_kl_divergence + step_kl
                    
                    if compute_entropy:
                        entropy_vocab = -(torch.exp(model_log_probs) * model_log_probs).sum(dim=2)
                        step_entropy = (entropy_vocab * mask * p_unmask).sum(dim=1) / self.seq_length
                        total_entropy = total_entropy + step_entropy

                tokens = torch.where(is_updatable, new_samples, tokens)
                
        returns = [tokens]
        if compute_kl:
            returns.append(total_kl_divergence)
        if compute_entropy:
            returns.append(total_entropy)

        if len(returns) == 1:
            return tokens
        return tuple(returns)
