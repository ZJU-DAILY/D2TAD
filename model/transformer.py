import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except Exception:  # pragma: no cover - optional fallback for CPU/lightweight envs.
    flash_attn_varlen_qkvpacked_func = None

try:
    from huggingface_hub import PyTorchModelHubMixin
except Exception:  # pragma: no cover - optional dependency for local-only use.
    class PyTorchModelHubMixin:
        pass

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = dim // n_heads
        self.dropout = float(dropout)

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, c):
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        qkv = self.attn_qkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )

        if qkv.is_cuda and flash_attn_varlen_qkvpacked_func is not None:
            qkv_flat = rearrange(qkv, "b s ... -> (b s) ...")
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=qkv.device,
            )
            attn = flash_attn_varlen_qkvpacked_func(
                qkv_flat, cu_seqlens, seq_len, 0.0, causal=False
            )
            attn = rearrange(attn, "(b s) h d -> b s (h d)", b=batch_size)
        else:
            q, k, v = qkv.unbind(dim=2)
            q = rearrange(q, "b s h d -> b h s d")
            k = rearrange(k, "b s h d -> b h s d")
            v = rearrange(v, "b s h d -> b h s d")
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                is_causal=False,
            )
            attn = rearrange(attn, "b h s d -> b s (h d)")
        x = bias_dropout_scale_fn(self.attn_out(attn), None, gate_msa, x_skip, self.dropout)

        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None,
            gate_mlp,
            x,
            self.dropout,
        )
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def get_embedding_weight(self):
        return self.embedding

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        return self.linear(x)


class SEDD(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        graph_type = str(config.graph.type).strip().lower()
        if graph_type not in {"absorb", "uniform"}:
            raise ValueError("SEDD submission build supports only absorb and uniform graphs.")

        self.absorb = graph_type == "absorb"
        vocab_size = int(config.tokens) + (1 if self.absorb else 0)

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        self.rotary_emb = rotary.Rotary(
            config.model.hidden_size // config.model.n_heads
        )

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                )
                for _ in range(config.model.n_blocks)
            ]
        )
        self.output_layer = DDitFinalLayer(
            config.model.hidden_size, vocab_size, config.model.cond_dim
        )
        self.scale_by_sigma = bool(config.model.scale_by_sigma)

    def forward(self, indices, sigma, **_ignored):
        if indices.dtype.is_floating_point or indices.dim() == 3:
            if indices.dim() != 3:
                raise ValueError("Soft input expects shape [B, L, V].")
            weight = self.vocab_embed.get_embedding_weight().to(indices.dtype)
            if indices.size(-1) != weight.size(0):
                raise ValueError(
                    f"Soft input vocab mismatch: got {indices.size(-1)} vs {weight.size(0)}."
                )
            x = torch.matmul(indices, weight)
        else:
            x = self.vocab_embed(indices)

        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)
        amp_context = (
            torch.cuda.amp.autocast(dtype=torch.bfloat16)
            if x.is_cuda
            else nullcontext()
        )
        with amp_context:
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c)
            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            if not self.absorb:
                raise ValueError("scale_by_sigma is only configured for absorb graph.")
            esigm1_log = torch.where(
                sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1
            ).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)

        if indices.dtype.is_floating_point or indices.dim() == 3:
            scatter_idx = indices.argmax(dim=-1)
        else:
            scatter_idx = indices
        x = torch.scatter(x, -1, scatter_idx[..., None], torch.zeros_like(x[..., :1]))
        return x
