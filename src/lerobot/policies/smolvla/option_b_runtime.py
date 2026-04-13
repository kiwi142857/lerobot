from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class OptionBDraft(nn.Module):
    def __init__(
        self,
        *,
        input_layers: int = 32,
        compressed_layers: int = 8,
        kv_feature_dim: int = 640,
        ctx_dim: int = 256,
        num_queries: int = 8,
        state_hidden_dim: int = 512,
        output_dim: int = 50 * 32,
    ) -> None:
        super().__init__()
        self.compressed_layers = compressed_layers
        self.ctx_dim = ctx_dim
        self.num_queries = num_queries

        self.layer_logits = nn.Parameter(torch.zeros(compressed_layers, input_layers))
        self.kv_proj = nn.Linear(kv_feature_dim, ctx_dim)
        self.query_tokens = nn.Parameter(torch.randn(num_queries, ctx_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=ctx_dim,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_ln_q = nn.LayerNorm(ctx_dim)
        self.cross_ln_kv = nn.LayerNorm(ctx_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(ctx_dim, 4 * ctx_dim),
            nn.GELU(),
            nn.Linear(4 * ctx_dim, ctx_dim),
        )
        self.cross_ffn_ln = nn.LayerNorm(ctx_dim)

        state_input_dim = (50 * 32) * 2 + 1
        self.state_mlp = nn.Sequential(
            nn.Linear(state_input_dim, state_hidden_dim),
            nn.GELU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.GELU(),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(state_hidden_dim + ctx_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )

    def forward(
        self,
        *,
        x0: torch.Tensor,
        v1: torch.Tensor,
        t1: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        kv_cache_k: torch.Tensor,
        kv_cache_v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x0.shape[0]

        kv_k = kv_cache_k.reshape(batch_size, kv_cache_k.shape[1], kv_cache_k.shape[2], -1)
        kv_v = kv_cache_v.reshape(batch_size, kv_cache_v.shape[1], kv_cache_v.shape[2], -1)
        kv = torch.cat([kv_k, kv_v], dim=-1).to(dtype=x0.dtype)

        mix = torch.softmax(self.layer_logits, dim=-1)
        kv = torch.einsum("bltf,ol->botf", kv, mix)
        kv = self.kv_proj(kv)
        kv = self.cross_ln_kv(kv)

        kv_seq = kv.reshape(batch_size, self.compressed_layers * kv.shape[2], self.ctx_dim)
        valid = prefix_pad_masks[:, None, :].expand(batch_size, self.compressed_layers, prefix_pad_masks.shape[1])
        key_padding_mask = ~valid.reshape(batch_size, self.compressed_layers * prefix_pad_masks.shape[1])

        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(
            query=self.cross_ln_q(queries),
            key=kv_seq,
            value=kv_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.cross_ffn(self.cross_ffn_ln(queries))
        ctx_vec = queries.mean(dim=1)

        state_vec = torch.cat(
            [
                x0.reshape(batch_size, -1),
                v1.reshape(batch_size, -1),
                t1.reshape(batch_size, -1),
            ],
            dim=-1,
        )
        state_vec = self.state_mlp(state_vec)
        fused = torch.cat([state_vec, ctx_vec], dim=-1)
        return self.fuse_mlp(fused).reshape(batch_size, 50, 32)


def load_option_b_runtime(
    draft_path: str,
    norm_stats_path: str | None = None,
) -> dict:
    draft_path_obj = Path(draft_path)
    if norm_stats_path is None:
        norm_stats_path_obj = draft_path_obj.with_name("norm_stats.pt")
    else:
        norm_stats_path_obj = Path(norm_stats_path)

    ckpt = torch.load(draft_path_obj, map_location="cpu", weights_only=False)
    norm_stats = torch.load(norm_stats_path_obj, map_location="cpu", weights_only=False)

    model = OptionBDraft()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "norm_stats": norm_stats,
        "device": "cpu",
        "draft_path": str(draft_path_obj),
        "norm_stats_path": str(norm_stats_path_obj),
    }
