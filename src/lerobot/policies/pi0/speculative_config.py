"""
Speculative decoding configuration for π0.

Usage (in your eval script or notebook):
    from lerobot.policies.pi0.speculative_config import SpeculativeConfig, load_spec_config
    policy.model.spec_config = load_spec_config(SpeculativeConfig(
        method="mlp_scheme_g",
        K=3,
        threshold=0.1,
        draft_head_path="/mnt/nvme3/cqw/spec-vla/eval_outputs/draft_head/best_draft_head.pt",
    ), device="cuda:0")

Set policy.model.spec_config = None to disable (default).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# MLP draft head (must match train_draft_head.py)
# ─────────────────────────────────────────────
class DraftHead(nn.Module):
    def __init__(self, input_dim: int = 1057, hidden_dim: int = 512, output_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────
@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    method:
        None              — disabled (standard inference)
        "mlp_scheme_g"    — MLP draft head + scheme-G linear-extrapolation verification
        "naive_scheme_g"  — prev-step v_t as draft + scheme-G verification (no model needed)
        "mlp_no_verify"   — MLP draft head, accept unconditionally (no target forward needed)
                            Fastest option; trades off some accuracy for maximum speedup.

    K:          number of steps to skip when draft is accepted
    threshold:  relative-error threshold for scheme-G acceptance (ignored for mlp_no_verify)
    draft_head_path: path to best_draft_head.pt (only for mlp_* methods)
    """
    method: Optional[str] = None          # None = disabled
    K: int = 3
    threshold: float = 0.1
    draft_head_path: str = (
        "/mnt/nvme3/cqw/spec-vla/eval_outputs/draft_head/best_draft_head.pt"
    )


# ─────────────────────────────────────────────
# Loader — returns a ready-to-use runtime dict
# ─────────────────────────────────────────────
def load_spec_config(cfg: SpeculativeConfig, device: str = "cuda:0") -> Optional[dict]:
    """Load draft head weights and norm stats; return runtime dict for sample_actions.

    Returns None if cfg.method is None (disables speculative decoding).
    """
    if cfg.method is None:
        return None

    runtime = {
        "method":    cfg.method,
        "K":         cfg.K,
        "threshold": cfg.threshold,
    }

    if cfg.method == "mlp_scheme_g":
        ckpt = torch.load(cfg.draft_head_path, map_location="cpu", weights_only=False)
        model = DraftHead()
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()
        runtime["model"] = model

        ns = ckpt["norm_stats"]
        runtime["norm_stats"] = {k: v.to(device) for k, v in ns.items()}

    elif cfg.method == "naive_scheme_g":
        # No model needed; draft = previous step's v_t
        runtime["model"] = None
        runtime["norm_stats"] = None

    elif cfg.method == "mlp_no_verify":
        ckpt = torch.load(cfg.draft_head_path, map_location="cpu", weights_only=False)
        model = DraftHead()
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()
        runtime["model"] = model

        ns = ckpt["norm_stats"]
        runtime["norm_stats"] = {k: v.to(device) for k, v in ns.items()}

    else:
        raise ValueError(f"Unknown speculative method: {cfg.method!r}")

    return runtime
