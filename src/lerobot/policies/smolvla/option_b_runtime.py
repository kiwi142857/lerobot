from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ContextCrossBlock(nn.Module):
    def __init__(self, ctx_dim: int, num_heads: int, ffn_multiplier: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=ctx_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ln_q = nn.LayerNorm(ctx_dim)
        self.ffn = nn.Sequential(
            nn.Linear(ctx_dim, ffn_multiplier * ctx_dim),
            nn.GELU(),
            nn.Linear(ffn_multiplier * ctx_dim, ctx_dim),
        )
        self.ffn_ln = nn.LayerNorm(ctx_dim)

    def forward(
        self,
        queries: torch.Tensor,
        kv_seq: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            query=self.ln_q(queries),
            key=kv_seq,
            value=kv_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        queries = queries + attn_out
        queries = queries + self.ffn(self.ffn_ln(queries))
        return queries


class OptionBDraft(nn.Module):
    def __init__(
        self,
        *,
        input_layers: int = 32,
        compressed_layers: int = 8,
        kv_feature_dim: int = 640,
        ctx_dim: int = 256,
        num_queries: int = 8,
        cross_num_heads: int = 8,
        cross_ffn_multiplier: int = 4,
        cross_block_count: int = 1,
        state_hidden_dim: int = 512,
        fuse_hidden_dim: int = 1024,
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
            num_heads=cross_num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_ln_q = nn.LayerNorm(ctx_dim)
        self.cross_ln_kv = nn.LayerNorm(ctx_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(ctx_dim, cross_ffn_multiplier * ctx_dim),
            nn.GELU(),
            nn.Linear(cross_ffn_multiplier * ctx_dim, ctx_dim),
        )
        self.cross_ffn_ln = nn.LayerNorm(ctx_dim)
        self.extra_cross_blocks = nn.ModuleList(
            [
                ContextCrossBlock(
                    ctx_dim=ctx_dim,
                    num_heads=cross_num_heads,
                    ffn_multiplier=cross_ffn_multiplier,
                )
                for _ in range(max(0, cross_block_count - 1))
            ]
        )

        state_input_dim = (50 * 32) * 2 + 1
        self.state_mlp = nn.Sequential(
            nn.Linear(state_input_dim, state_hidden_dim),
            nn.GELU(),
            nn.Linear(state_hidden_dim, state_hidden_dim),
            nn.GELU(),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(state_hidden_dim + ctx_dim, fuse_hidden_dim),
            nn.GELU(),
            nn.Linear(fuse_hidden_dim, output_dim),
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
        for block in self.extra_cross_blocks:
            queries = block(queries, kv_seq, key_padding_mask)
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


def infer_option_b_model_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, int]:
    layer_logits = state_dict["layer_logits"]
    kv_proj_weight = state_dict["kv_proj.weight"]
    query_tokens = state_dict["query_tokens"]
    state_mlp0_weight = state_dict["state_mlp.0.weight"]
    fuse_mlp0_weight = state_dict["fuse_mlp.0.weight"]
    fuse_mlp2_weight = state_dict["fuse_mlp.2.weight"]
    cross_ffn0_weight = state_dict["cross_ffn.0.weight"]
    extra_cross_blocks = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("extra_cross_blocks.")
        }
    )
    ctx_dim = int(kv_proj_weight.shape[0])
    return {
        "input_layers": int(layer_logits.shape[1]),
        "compressed_layers": int(layer_logits.shape[0]),
        "kv_feature_dim": int(kv_proj_weight.shape[1]),
        "ctx_dim": ctx_dim,
        "num_queries": int(query_tokens.shape[0]),
        "cross_num_heads": 8,
        "cross_ffn_multiplier": int(cross_ffn0_weight.shape[0] // ctx_dim),
        "cross_block_count": 1 + len(extra_cross_blocks),
        "state_hidden_dim": int(state_mlp0_weight.shape[0]),
        "fuse_hidden_dim": int(fuse_mlp0_weight.shape[0]),
        "output_dim": int(fuse_mlp2_weight.shape[0]),
    }


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

    model_config = ckpt.get("model_config")
    if model_config is None:
        model_config = infer_option_b_model_config_from_state_dict(ckpt["model_state_dict"])

    model = OptionBDraft(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "norm_stats": norm_stats,
        "device": "cpu",
        "draft_path": str(draft_path_obj),
        "norm_stats_path": str(norm_stats_path_obj),
        "model_config": model_config,
    }
