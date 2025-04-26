# Qwen2 model package initialization

import os
import re
import torch
from transformers import AutoModelForCausalLM
from zeroband.config import Config
from zeroband.models.qwen2.model import ModelArgs, Qwen2ForCausalLM

# Qwen2 7B default configuration
qwen2_configs = {
    "Qwen/Qwen2.5-0.5B": ModelArgs(
        dim=896,
        n_layers=24,
        n_heads=14,
        n_kv_heads=2,
        rope_theta=1000000,
    ),
    "7B": ModelArgs(
        dim=3584,
        n_layers=28,
        n_heads=28,
        n_kv_heads=4,
        rope_theta=1000000,
    )
}

__all__ = ["Qwen2ForCausalLM"]

def get_model(
    config: Config,
    vocab_size: int = None
) -> tuple[Qwen2ForCausalLM, ModelArgs]:
    """get the Qwen2ForCausalLM model"""

    if config.type_model == "qwen2":
        model_config = qwen2_configs[config.name_model]
    else:
        raise ValueError(f"Model type {config.type_model} not supported")

    # Override vocab size if provided
    if vocab_size is not None:
        model_config.vocab_size = vocab_size

    return Qwen2ForCausalLM(model_config), model_config

def _convert_key(hf_key: str, tensor) -> list[tuple[str, torch.Tensor]]:
    """
    Map HuggingFace parameter names to Prime names.
    Handles the only structural difference: fused gate_up_proj.
    Everything else is 1‑to‑1.
    """
    # strip leading "model."
    if hf_key.startswith("model."):
        hf_key = hf_key[len("model.") :]

    # Qwen‑2 HF uses fused gate_up_proj → split into 2 tensors
    m = re.match(r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)", hf_key)
    if m:
        layer_idx, wb = m.groups()
        gate, up = torch.chunk(tensor, 2, dim=0)
        base = f"model.layers.{layer_idx}.mlp"
        return [
            (f"{base}.gate_proj.{wb}", gate.contiguous()),
            (f"{base}.up_proj.{wb}", up.contiguous()),
        ]

    # everything else: prepend "model."
    return [(f"model.{hf_key}", tensor)]


@torch.no_grad()
def load_hf_weights(
    model: Qwen2ForCausalLM,
    hf_repo: str | os.PathLike,
    dtype: torch.dtype | None = None,
    device: str = "cpu",
) -> Qwen2ForCausalLM:
    """
    Load weights from a HuggingFace Qwen‑2 checkpoint into a Prime model.

    Example:
        prime_model, _ = get_model(cfg, tokenizer.vocab_size)
        load_hf_weights(prime_model, "Qwen/Qwen2-7B-Chat")
    """
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_repo, torch_dtype=dtype or model.lm_head.weight.dtype, device_map=device
    )
    new_state = {}
    for k, v in hf_model.state_dict().items():
        for new_k, new_v in _convert_key(k, v):
            new_state[new_k] = new_v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    return model.eval()
