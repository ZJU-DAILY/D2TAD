import argparse
import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader

import data
import graph_lib
import noise_lib
import utils
from load_model import load_model_local
from model import rotary
from model.transformer import DDitFinalLayer, LayerNorm, modulate_fused


def _maybe_cuda_autocast(tensor):
    if tensor.is_cuda:
        return torch.cuda.amp.autocast(dtype=torch.bfloat16)
    return nullcontext()


class CausalDDiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        cond_dim,
        mlp_ratio=4,
        dropout=0.1,
        chunk_size=1,
        max_cache_len=None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.dim // self.n_heads
        self.dropout = float(dropout)
        self.chunk_size = max(1, int(chunk_size))
        self.max_cache_len = None if max_cache_len is None else int(max_cache_len)

        self.norm1 = LayerNorm(self.dim)
        self.attn_qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.attn_out = nn.Linear(self.dim, self.dim, bias=False)

        self.norm2 = LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, mlp_ratio * self.dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * self.dim, self.dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * self.dim, bias=True)

    def _parse_cache(self, past_cache):
        if past_cache is None:
            return None, None, 0
        if len(past_cache) == 3:
            cache_k, cache_v, cache_len = past_cache
            return cache_k, cache_v, int(cache_len)
        cache_k, cache_v = past_cache
        return cache_k, cache_v, int(cache_k.size(2))

    def _ensure_cache_capacity(
        self, cache_k, cache_v, batch_size, total_len, dtype, device
    ):
        if cache_k is None or cache_v is None:
            alloc_len = max(
                int(total_len),
                int(self.max_cache_len) if self.max_cache_len is not None else 0,
            )
            cache_k = torch.empty(
                (batch_size, self.n_heads, alloc_len, self.head_dim),
                device=device,
                dtype=dtype,
            )
            cache_v = torch.empty_like(cache_k)
            return cache_k, cache_v
        if cache_k.size(2) >= total_len:
            return cache_k, cache_v
        new_cap = max(int(cache_k.size(2) * 2), int(total_len))
        new_k = torch.empty(
            (batch_size, self.n_heads, new_cap, self.head_dim),
            device=device,
            dtype=dtype,
        )
        new_v = torch.empty_like(new_k)
        old_len = int(cache_k.size(2))
        new_k[:, :, :old_len].copy_(cache_k)
        new_v[:, :, :old_len].copy_(cache_v)
        return new_k, new_v

    @classmethod
    def from_teacher(cls, teacher_block, chunk_size=1, max_cache_len=None):
        dim = teacher_block.attn_out.weight.shape[0]
        n_heads = int(teacher_block.n_heads)
        cond_dim = teacher_block.adaLN_modulation.in_features
        mlp_ratio = teacher_block.mlp[0].out_features // dim
        dropout = float(getattr(teacher_block, "dropout", 0.0))

        block = cls(
            dim=dim,
            n_heads=n_heads,
            cond_dim=cond_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            chunk_size=chunk_size,
            max_cache_len=max_cache_len,
        )
        block.norm1.load_state_dict(copy.deepcopy(teacher_block.norm1.state_dict()))
        block.attn_qkv.load_state_dict(
            copy.deepcopy(teacher_block.attn_qkv.state_dict())
        )
        block.attn_out.load_state_dict(
            copy.deepcopy(teacher_block.attn_out.state_dict())
        )
        block.norm2.load_state_dict(copy.deepcopy(teacher_block.norm2.state_dict()))
        block.mlp.load_state_dict(copy.deepcopy(teacher_block.mlp.state_dict()))
        block.adaLN_modulation.load_state_dict(
            copy.deepcopy(teacher_block.adaLN_modulation.state_dict())
        )
        return block

    def _build_block_attn_mask(self, seq_len, device, start_position=0):
        positions_q = torch.arange(
            start_position, start_position + seq_len, device=device, dtype=torch.long
        )
        positions_k = torch.arange(
            start_position, start_position + seq_len, device=device, dtype=torch.long
        )
        if self.chunk_size <= 1:
            return positions_k[None, :] <= positions_q[:, None]
        chunk_ends = ((positions_q // self.chunk_size) + 1) * self.chunk_size
        return positions_k[None, :] < chunk_ends[:, None]

    def forward(self, x, rotary_cos_sin, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        x_norm = self.norm1(x)
        x_mod = modulate_fused(x_norm, shift_msa, scale_msa)

        qkv = self.attn_qkv(x_mod)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q, k, v = qkv.unbind(dim=2)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")
        use_block_mask = attn_mask is not None or self.chunk_size > 1
        if use_block_mask:
            if attn_mask is None:
                attn_mask = self._build_block_attn_mask(x.size(1), x.device)[None, None]
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        x = x_skip + gate_msa * F.dropout(
            self.attn_out(attn_out), p=self.dropout, training=self.training
        )

        x_norm = self.norm2(x)
        mlp_out = self.mlp(modulate_fused(x_norm, shift_mlp, scale_mlp))
        x = x + gate_mlp * F.dropout(mlp_out, p=self.dropout, training=self.training)
        return x

    def forward_incremental(
        self, x, rotary_cos_sin, c, past_cache=None, update_cache=False
    ):
        if x.size(1) != 1:
            raise ValueError(
                "forward_incremental currently expects a single token step."
            )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        x_norm = self.norm1(x)
        x_mod = modulate_fused(x_norm, shift_msa, scale_msa)

        qkv = self.attn_qkv(x_mod)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q, k, v = qkv.unbind(dim=2)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        cache_k, cache_v, cache_len = self._parse_cache(past_cache)
        step_len = int(k.size(2))
        total_len = cache_len + step_len
        if cache_k is None and not update_cache:
            full_k = k
            full_v = v
        else:
            cache_k, cache_v = self._ensure_cache_capacity(
                cache_k,
                cache_v,
                batch_size=int(k.size(0)),
                total_len=total_len,
                dtype=k.dtype,
                device=k.device,
            )
            cache_k[:, :, cache_len:total_len].copy_(k)
            cache_v[:, :, cache_len:total_len].copy_(v)
            full_k = cache_k[:, :, :total_len]
            full_v = cache_v[:, :, :total_len]

        attn_out = F.scaled_dot_product_attention(
            q, full_k, full_v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        x = x_skip + gate_msa * self.attn_out(attn_out)

        x_norm = self.norm2(x)
        mlp_out = self.mlp(modulate_fused(x_norm, shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out
        next_cache = (cache_k, cache_v, total_len) if update_cache else past_cache
        return x, next_cache

    def forward_incremental_block(
        self,
        x,
        rotary_cos_sin,
        c,
        past_cache=None,
        update_cache=False,
        start_position=0,
    ):
        if x.dim() != 3 or x.size(1) <= 0:
            raise ValueError(
                "forward_incremental_block expects shape [B, S, D] with S > 0."
            )

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        )

        x_skip = x
        x_norm = self.norm1(x)
        x_mod = modulate_fused(x_norm, shift_msa, scale_msa)

        qkv = self.attn_qkv(x_mod)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        q, k, v = qkv.unbind(dim=2)
        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        cache_k, cache_v, past_len = self._parse_cache(past_cache)
        block_len = int(q.size(2))
        total_len = past_len + block_len
        if cache_k is None and not update_cache:
            full_k = k
            full_v = v
        else:
            cache_k, cache_v = self._ensure_cache_capacity(
                cache_k,
                cache_v,
                batch_size=int(k.size(0)),
                total_len=total_len,
                dtype=k.dtype,
                device=k.device,
            )
            cache_k[:, :, past_len:total_len].copy_(k)
            cache_v[:, :, past_len:total_len].copy_(v)
            full_k = cache_k[:, :, :total_len]
            full_v = cache_v[:, :, :total_len]

        allowed_mask = torch.ones(
            (block_len, past_len + block_len), device=q.device, dtype=torch.bool
        )
        current_mask = self._build_block_attn_mask(
            block_len, q.device, start_position=int(start_position)
        )
        if start_position == past_len:
            allowed_mask[:, past_len:] = current_mask
        else:
            allowed_mask[:, past_len:] = current_mask
        attn_out = F.scaled_dot_product_attention(
            q,
            full_k,
            full_v,
            attn_mask=allowed_mask[None, None],
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        x = x_skip + gate_msa * self.attn_out(attn_out)

        x_norm = self.norm2(x)
        mlp_out = self.mlp(modulate_fused(x_norm, shift_mlp, scale_mlp))
        x = x + gate_mlp * mlp_out
        next_cache = (cache_k, cache_v, total_len) if update_cache else past_cache
        return x, next_cache


class CausalSEDDStudent(nn.Module):
    def __init__(self, teacher=None, cfg=None, chunk_size=None):
        super().__init__()
        if teacher is None and cfg is None:
            raise ValueError("Either teacher or cfg must be provided.")
        if teacher is not None:
            cfg = copy.deepcopy(teacher.config)
        else:
            cfg = copy.deepcopy(cfg)

        if chunk_size is None:
            chunk_size = int(getattr(cfg.model, "student_chunk_size", 1))
        chunk_size = max(1, int(chunk_size))
        setattr(cfg.model, "student_chunk_size", chunk_size)

        self.config = cfg
        self.chunk_size = chunk_size
        self.max_cache_len = int(cfg.model.length)
        if teacher is not None:
            self.absorb = teacher.absorb
            self.vocab_embed = copy.deepcopy(teacher.vocab_embed)
            self.sigma_map = copy.deepcopy(teacher.sigma_map)
        else:
            from model.transformer import EmbeddingLayer, TimestepEmbedder

            self.absorb = cfg.graph.type == "absorb"
            vocab_size = int(cfg.tokens) + (1 if self.absorb else 0)
            self.vocab_embed = EmbeddingLayer(cfg.model.hidden_size, vocab_size)
            self.sigma_map = TimestepEmbedder(cfg.model.cond_dim)
        self.rotary_emb = rotary.Rotary(cfg.model.hidden_size // cfg.model.n_heads)
        if teacher is not None:
            self.blocks = nn.ModuleList(
                [
                    CausalDDiTBlock.from_teacher(
                        block,
                        chunk_size=chunk_size,
                        max_cache_len=self.max_cache_len,
                    )
                    for block in teacher.blocks
                ]
            )
            self.output_layer = copy.deepcopy(teacher.output_layer)
        else:
            self.blocks = nn.ModuleList(
                [
                    CausalDDiTBlock(
                        dim=cfg.model.hidden_size,
                        n_heads=cfg.model.n_heads,
                        cond_dim=cfg.model.cond_dim,
                        mlp_ratio=4,
                        dropout=cfg.model.dropout,
                        chunk_size=chunk_size,
                        max_cache_len=self.max_cache_len,
                    )
                    for _ in range(cfg.model.n_blocks)
                ]
            )
            vocab_size = int(cfg.tokens) + (1 if self.absorb else 0)
            self.output_layer = DDitFinalLayer(
                cfg.model.hidden_size, vocab_size, cfg.model.cond_dim
            )
        self.scale_by_sigma = bool(cfg.model.scale_by_sigma)

    def _embed_indices(self, indices):
        if indices.dim() == 1:
            indices = indices[:, None]
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
        return x, indices

    def _sigma_cond(self, sigma):
        return F.silu(self.sigma_map(sigma))

    def _rotary_slice(self, start_pos, seq_len, device, dtype):
        inv_freq = self.rotary_emb.inv_freq.to(device=device)
        positions = torch.arange(start_pos, start_pos + seq_len, device=device).type_as(
            inv_freq
        )
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1).to(dtype=dtype)
        sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1).to(dtype=dtype)
        cos[:, :, 2].fill_(1.0)
        sin[:, :, 2].fill_(0.0)
        return cos, sin

    def forward(self, indices, sigma, valid_mask=None):
        x, indices = self._embed_indices(indices)
        c = self._sigma_cond(sigma)
        rotary_cos_sin = self.rotary_emb(x)
        attn_mask = None
        if self.chunk_size > 1:
            base_mask = self.blocks[0]._build_block_attn_mask(x.size(1), x.device)[
                None, None
            ]
            if valid_mask is not None:
                valid_key_mask = valid_mask.to(device=x.device, dtype=torch.bool)[
                    :, None, None, :
                ]
                attn_mask = base_mask & valid_key_mask
            else:
                attn_mask = base_mask

        with _maybe_cuda_autocast(x):
            for block in self.blocks:
                x = block(x, rotary_cos_sin, c, attn_mask=attn_mask)
            x = self.output_layer(x, c)

        if self.scale_by_sigma:
            if not self.absorb:
                raise ValueError("scale_by_sigma is only configured for absorb graph.")
            esigm1_log = (
                torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                .log()
                .to(x.dtype)[:, None, None]
            )
            x = x - esigm1_log - math.log(x.shape[-1] - 1)

        if indices.dtype.is_floating_point or indices.dim() == 3:
            scatter_idx = indices.argmax(dim=-1)
        else:
            scatter_idx = indices
        x = torch.scatter(x, -1, scatter_idx[..., None], torch.zeros_like(x[..., :1]))
        return x

    def forward_incremental(
        self,
        indices,
        sigma,
        position,
        kv_cache=None,
        update_cache=False,
        compute_logits=True,
    ):
        x, indices = self._embed_indices(indices)
        if x.size(1) != 1:
            raise ValueError("forward_incremental expects a single token step.")
        c = self._sigma_cond(sigma)
        rotary_cos_sin = self._rotary_slice(int(position), 1, x.device, x.dtype)

        next_cache = [] if update_cache else kv_cache
        with _maybe_cuda_autocast(x):
            for block_idx, block in enumerate(self.blocks):
                past_cache = None if kv_cache is None else kv_cache[block_idx]
                x, block_cache = block.forward_incremental(
                    x,
                    rotary_cos_sin,
                    c,
                    past_cache=past_cache,
                    update_cache=update_cache,
                )
                if update_cache:
                    next_cache.append(block_cache)
            logits = None
            if compute_logits:
                logits = self.output_layer(x, c)

        if not compute_logits:
            return None, next_cache

        if self.scale_by_sigma:
            if not self.absorb:
                raise ValueError("scale_by_sigma is only configured for absorb graph.")
            esigm1_log = (
                torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                .log()
                .to(logits.dtype)[:, None, None]
            )
            logits = logits - esigm1_log - math.log(logits.shape[-1] - 1)

        if indices.dtype.is_floating_point or indices.dim() == 3:
            scatter_idx = indices.argmax(dim=-1)
        else:
            scatter_idx = indices
        logits = torch.scatter(
            logits, -1, scatter_idx[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits, next_cache

    def forward_incremental_block(
        self,
        indices,
        sigma,
        start_position,
        kv_cache=None,
        update_cache=False,
        compute_logits=True,
    ):
        x, indices = self._embed_indices(indices)
        if x.dim() != 3 or x.size(1) <= 0:
            raise ValueError("forward_incremental_block expects input shape [B, S].")
        c = self._sigma_cond(sigma)
        rotary_cos_sin = self._rotary_slice(
            int(start_position), x.size(1), x.device, x.dtype
        )

        next_cache = [] if update_cache else kv_cache
        with _maybe_cuda_autocast(x):
            for block_idx, block in enumerate(self.blocks):
                past_cache = None if kv_cache is None else kv_cache[block_idx]
                x, block_cache = block.forward_incremental_block(
                    x,
                    rotary_cos_sin,
                    c,
                    past_cache=past_cache,
                    update_cache=update_cache,
                    start_position=int(start_position),
                )
                if update_cache:
                    next_cache.append(block_cache)
            logits = None
            if compute_logits:
                logits = self.output_layer(x, c)

        if not compute_logits:
            return None, next_cache

        if self.scale_by_sigma:
            if not self.absorb:
                raise ValueError("scale_by_sigma is only configured for absorb graph.")
            esigm1_log = (
                torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                .log()
                .to(logits.dtype)[:, None, None]
            )
            logits = logits - esigm1_log - math.log(logits.shape[-1] - 1)

        if indices.dtype.is_floating_point or indices.dim() == 3:
            scatter_idx = indices.argmax(dim=-1)
        else:
            scatter_idx = indices
        logits = torch.scatter(
            logits, -1, scatter_idx[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits, next_cache


def build_loader(cfg, split, batch_size, shuffle, num_workers):
    dataset = data._build_proto_dataset(cfg, split=split, block_size=cfg.model.length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=shuffle,
    )


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def sample_sigma(noise, batch_size, device, eps):
    t = torch.rand(batch_size, device=device)
    t = t * (1.0 - eps) + eps
    sigma, _ = noise(t)
    return sigma


def build_online_suffix_mask(valid_mask, min_ratio=0.0, max_ratio=0.5):
    if valid_mask is None:
        return None
    valid_mask = valid_mask.bool()
    batch_size, seq_len = valid_mask.shape
    valid_lens = valid_mask.sum(dim=1)
    if batch_size == 0:
        return valid_mask
    min_ratio = float(min_ratio)
    max_ratio = float(max_ratio)
    if max_ratio < min_ratio:
        raise ValueError("prefix_ratio_max must be >= prefix_ratio_min.")
    ratios = torch.empty(batch_size, device=valid_mask.device).uniform_(
        min_ratio, max_ratio
    )
    obs_lens = torch.round(valid_lens.float() * ratios).long()
    obs_lens = torch.minimum(obs_lens, (valid_lens - 1).clamp_min(1))
    obs_lens = obs_lens.clamp_min(1)
    positions = torch.arange(seq_len, device=valid_mask.device)[None, :]
    suffix_mask = valid_mask & (positions >= obs_lens[:, None])
    empty_rows = suffix_mask.sum(dim=1) == 0
    if empty_rows.any():
        fallback_pos = (valid_lens[empty_rows] - 1).clamp_min(0)
        suffix_mask[empty_rows] = False
        suffix_mask[empty_rows, fallback_pos] = True
    return suffix_mask


def compute_teacher_hardness_weights(
    teacher_logits,
    target_tokens,
    active_mask,
    temperature=1.0,
    hardness_alpha=0.0,
):
    if active_mask is None:
        active_mask = teacher_logits.new_ones(
            teacher_logits.shape[:2], dtype=torch.bool
        )
    if float(hardness_alpha) <= 0.0:
        return active_mask.to(dtype=teacher_logits.dtype)
    t = float(temperature)
    teacher_log_prob = F.log_softmax(teacher_logits / t, dim=-1)
    token_nll = -teacher_log_prob.gather(
        dim=-1, index=target_tokens.long().unsqueeze(-1)
    ).squeeze(-1)
    mask = active_mask.bool()
    masked_nll = token_nll.masked_fill(~mask, 0.0)
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    mean = masked_nll.sum(dim=1, keepdim=True) / counts
    centered = (token_nll - mean).masked_fill(~mask, 0.0)
    var = (centered * centered).sum(dim=1, keepdim=True) / counts
    std = var.sqrt().clamp_min(1e-6)
    z = ((token_nll - mean) / std).masked_fill(~mask, 0.0)
    weights = 1.0 + float(hardness_alpha) * torch.relu(z)
    return weights * mask.to(dtype=teacher_logits.dtype)


def masked_kl(student_logits, teacher_logits, valid_mask, temperature, weights=None):
    t = float(temperature)
    teacher_prob = F.softmax(teacher_logits / t, dim=-1)
    student_log_prob = F.log_softmax(student_logits / t, dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=-1) * (
        t * t
    )
    if weights is not None:
        weights = weights.to(kl.dtype)
    if valid_mask is not None:
        mask = valid_mask.to(kl.dtype)
        if weights is not None:
            mask = mask * weights
        denom = mask.sum().clamp_min(1.0)
        return (kl * mask).sum() / denom
    if weights is not None:
        denom = weights.sum().clamp_min(1.0)
        return (kl * weights).sum() / denom
    return kl.mean()


def compute_distill_loss(
    student_logits,
    teacher_logits,
    valid_mask,
    target_tokens,
    temperature,
    online_prefix_distill=False,
    prefix_ratio_min=0.0,
    prefix_ratio_max=0.5,
    hardness_alpha=0.0,
    prefix_condition_mode="mask",
    suffix_weight=1.0,
):
    active_mask = valid_mask
    suffix_mask = None
    if online_prefix_distill:
        suffix_mask = build_online_suffix_mask(
            valid_mask,
            min_ratio=prefix_ratio_min,
            max_ratio=prefix_ratio_max,
        )
        if str(prefix_condition_mode) == "mask":
            active_mask = suffix_mask
    weights = compute_teacher_hardness_weights(
        teacher_logits,
        target_tokens=target_tokens,
        active_mask=active_mask,
        temperature=temperature,
        hardness_alpha=hardness_alpha,
    )
    if (
        online_prefix_distill
        and suffix_mask is not None
        and str(prefix_condition_mode) == "boost"
        and float(suffix_weight) > 0.0
    ):
        weights = weights * (1.0 + suffix_mask.to(weights.dtype) * float(suffix_weight))
    loss = masked_kl(
        student_logits,
        teacher_logits,
        active_mask,
        temperature,
        weights=weights,
    )
    return loss, active_mask, weights, suffix_mask


@torch.no_grad()
def evaluate(
    student,
    teacher,
    graph,
    noise,
    loader,
    device,
    eps,
    temperature,
    max_batches,
    online_prefix_distill=False,
    prefix_ratio_min=0.0,
    prefix_ratio_max=0.5,
    hardness_alpha=0.0,
    prefix_condition_mode="mask",
    suffix_weight=1.0,
):
    student.eval()
    teacher.eval()
    total_loss = 0.0
    total_batches = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch(batch, device)
        x0 = batch["input_ids"]
        valid_mask = batch.get("valid_mask")
        sigma = sample_sigma(noise, x0.shape[0], device, eps)
        xt = graph.sample_transition(x0, sigma[:, None])
        with torch.inference_mode():
            teacher_logits = teacher(xt, sigma).float()
            student_logits = student(xt, sigma, valid_mask=valid_mask).float()
        loss, _, _, _ = compute_distill_loss(
            student_logits,
            teacher_logits,
            valid_mask,
            target_tokens=x0,
            temperature=temperature,
            online_prefix_distill=online_prefix_distill,
            prefix_ratio_min=prefix_ratio_min,
            prefix_ratio_max=prefix_ratio_max,
            hardness_alpha=hardness_alpha,
            prefix_condition_mode=prefix_condition_mode,
            suffix_weight=suffix_weight,
        )
        total_loss += float(loss.item())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling_eps", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--student_chunk_size",
        type=int,
        default=2,
        help="Chunk size for blockwise causal student training.",
    )
    parser.add_argument(
        "--online_prefix_distill",
        dest="online_prefix_distill",
        action="store_true",
        help="Train with random observed prefixes and only distill suffix positions.",
    )
    parser.add_argument(
        "--disable_online_prefix_distill",
        dest="online_prefix_distill",
        action="store_false",
        help="Disable online-prefix suffix distillation.",
    )
    parser.set_defaults(online_prefix_distill=True)
    parser.add_argument(
        "--prefix_ratio_min",
        type=float,
        default=0.0,
        help="Minimum observed-prefix ratio for online-prefix distillation.",
    )
    parser.add_argument(
        "--prefix_ratio_max",
        type=float,
        default=0.25,
        help="Maximum observed-prefix ratio for online-prefix distillation.",
    )
    parser.add_argument(
        "--hardness_alpha",
        type=float,
        default=0.01,
        help="Teacher-hardness reweighting strength based on clean-token NLL.",
    )
    parser.add_argument(
        "--prefix_condition_mode",
        type=str,
        default="mask",
        choices=["mask", "boost"],
        help="mask: only distill suffix positions; boost: keep all positions and upweight suffix positions.",
    )
    parser.add_argument(
        "--suffix_weight",
        type=float,
        default=1.0,
        help="Extra suffix weight used when prefix_condition_mode=boost.",
    )
    args = parser.parse_args()

    if int(args.student_chunk_size) <= 0:
        raise ValueError("student_chunk_size must be >= 1.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    teacher, graph, noise = load_model_local(args.teacher_model_path, device)
    cfg = teacher.config
    setattr(cfg.model, "student_chunk_size", int(args.student_chunk_size))

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in noise.parameters():
        p.requires_grad_(False)

    student = CausalSEDDStudent(teacher, chunk_size=int(args.student_chunk_size)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_loader = build_loader(
        cfg,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = build_loader(
        cfg,
        split="valid",
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_iter = iter(train_loader)

    teacher_root = Path(args.teacher_model_path).expanduser().resolve().parent.parent
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = teacher_root / "autoregressive_distill_runs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"teacher_model_path={args.teacher_model_path}\n")
        f.write(f"output_dir={output_dir}\n")
        f.write(f"device={device}\n")
        f.write(f"student_chunk_size={int(args.student_chunk_size)}\n")
        f.write(
            f"online_prefix_distill={bool(args.online_prefix_distill)} prefix_ratio_min={args.prefix_ratio_min} prefix_ratio_max={args.prefix_ratio_max} hardness_alpha={args.hardness_alpha} prefix_condition_mode={args.prefix_condition_mode} suffix_weight={args.suffix_weight}\n"
        )
        f.write(f"n_steps={args.n_steps} batch_size={args.batch_size} lr={args.lr}\n")

    print(f"teacher_model_path={args.teacher_model_path}")
    print(f"output_dir={output_dir}")
    print(f"device={device}")
    print(f"student_chunk_size={int(args.student_chunk_size)}")
    print(
        f"online_prefix_distill={bool(args.online_prefix_distill)} prefix_ratio_min={args.prefix_ratio_min} prefix_ratio_max={args.prefix_ratio_max} hardness_alpha={args.hardness_alpha} prefix_condition_mode={args.prefix_condition_mode} suffix_weight={args.suffix_weight}"
    )
    print(f"student_parameters={sum(p.numel() for p in student.parameters())}")

    start_time = time.time()
    student.train()
    for step in range(1, args.n_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = move_batch(batch, device)
        x0 = batch["input_ids"]
        valid_mask = batch.get("valid_mask")
        sigma = sample_sigma(noise, x0.shape[0], device, args.sampling_eps)
        xt = graph.sample_transition(x0, sigma[:, None])
        with torch.inference_mode():
            teacher_logits = teacher(xt, sigma).float()

        student_logits = student(xt, sigma, valid_mask=valid_mask).float()
        loss, active_mask, weights, suffix_mask = compute_distill_loss(
            student_logits,
            teacher_logits,
            valid_mask,
            target_tokens=x0,
            temperature=args.temperature,
            online_prefix_distill=args.online_prefix_distill,
            prefix_ratio_min=args.prefix_ratio_min,
            prefix_ratio_max=args.prefix_ratio_max,
            hardness_alpha=args.hardness_alpha,
            prefix_condition_mode=args.prefix_condition_mode,
            suffix_weight=args.suffix_weight,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            suffix_frac = 0.0
            if suffix_mask is not None:
                base_denom = valid_mask.sum().clamp_min(1).item()
                suffix_frac = float(suffix_mask.sum().item() / base_denom)
            mean_weight = (
                float(weights[active_mask].mean().item())
                if active_mask is not None and active_mask.any()
                else float(weights.mean().item())
            )
            msg = (
                f"step={step} train_kl={loss.item():.6f} "
                f"suffix_frac={suffix_frac:.4f} mean_weight={mean_weight:.4f} "
                f"elapsed={elapsed:.2f}s"
            )
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

        if step % args.eval_every == 0 or step == args.n_steps:
            eval_loss = evaluate(
                student,
                teacher,
                graph,
                noise,
                valid_loader,
                device,
                args.sampling_eps,
                args.temperature,
                args.eval_batches,
                online_prefix_distill=args.online_prefix_distill,
                prefix_ratio_min=args.prefix_ratio_min,
                prefix_ratio_max=args.prefix_ratio_max,
                hardness_alpha=args.hardness_alpha,
                prefix_condition_mode=args.prefix_condition_mode,
                suffix_weight=args.suffix_weight,
            )
            msg = f"step={step} valid_kl={eval_loss:.6f}"
            print(msg)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
            student.train()

        if step % args.save_every == 0 or step == args.n_steps:
            ckpt = {
                "step": step,
                "student": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "teacher_model_path": args.teacher_model_path,
                "config": cfg,
                "args": vars(args),
            }
            ckpt_path = ckpt_dir / f"student_step_{step}.pth"
            torch.save(ckpt, ckpt_path)
            print(f"saved_checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
