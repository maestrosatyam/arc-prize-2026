"""Model arms at matched parameter count: CPU arms about 0.7M (d_model 128), GPU arms about 21M (d_model 512)."""
from __future__ import annotations

from arcssm.models import ModelConfig

ARMS: dict[str, ModelConfig] = {
    "mamba": ModelConfig(arch="mamba", d_model=128, n_layers=6, d_state=32, expand=2, headdim=32, chunk=32),
    "transformer": ModelConfig(arch="transformer", d_model=128, n_layers=4, n_heads=4, mlp_mult=2.0),
    "gru": ModelConfig(arch="gru", d_model=128, n_layers=7),
    "hybrid": ModelConfig(arch="hybrid", d_model=128, n_layers=6, d_state=32, expand=2, headdim=32, chunk=32,
                          attn_layers=(3,), n_heads=4, mlp_mult=2.0),
}

# GPU arms (Kaggle L4), about 21-22M parameters, 9216-token RoPE table.
GPU_ARMS: dict[str, ModelConfig] = {
    "transformer_g": ModelConfig(arch="transformer", d_model=512, n_layers=8, n_heads=8, mlp_mult=2.0, max_len=9216),
    "mamba_g": ModelConfig(arch="mamba", d_model=512, n_layers=13, d_state=64, expand=2, headdim=64, chunk=64,
                           max_len=9216),
    "hybrid_g": ModelConfig(arch="hybrid", d_model=512, n_layers=12, d_state=64, expand=2, headdim=64, chunk=64,
                            attn_layers=(3, 9), n_heads=8, mlp_mult=2.0, max_len=9216),
}
ARMS.update(GPU_ARMS)
