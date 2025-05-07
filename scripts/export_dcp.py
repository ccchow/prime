#!/usr/bin/env python
# coding: utf-8
# Example Usage:
# python scripts/export_dcp.py @configs/10B/H100.toml --ckpt.path /data/intellect-1-step17000 --ckpt.resume /data/10b/step_17000/diloco_0

import torch
from typing import Literal
import torch.distributed.checkpoint as dcp
from zeroband.models.llama import get_model as get_model_llama
from zeroband.models.qwen2 import get_model as get_model_qwen2
from zeroband.config import resolve_env_vars
from zeroband.checkpoint import ModelWrapper
from zeroband.utils import get_module_signature
from zeroband.train import Config
from zeroband.utils.logger import get_logger
from pydantic_config import parse_argv
from transformers import AutoTokenizer
from transformers import LlamaConfig
from transformers import Qwen2Config
from transformers.generation import GenerationConfig
import math
import re
from pathlib import Path
from safetensors.torch import save_file
import json
import torch


class ExportConfig(Config):
    save_format: Literal["pt", "safetensors"] = "safetensors"
    torch_dtype: Literal["float32", "bfloat16"] = "float32"
    with_debug_automap: bool = False


def remap_keys_llama(k: str) -> str:
    """Maps ZeroBand keys to HuggingFace keys"""
    return ("model." if "output.weight" not in k else "") + k.replace("tok_embeddings", "embed_tokens").replace(
        "attention.wq", "self_attn.q_proj"
    ).replace("attention.wk", "self_attn.k_proj").replace("attention.wv", "self_attn.v_proj").replace(
        "attention.wo", "self_attn.o_proj"
    ).replace("attention_norm", "input_layernorm").replace("feed_forward.w3", "mlp.up_proj").replace(
        "feed_forward.w2", "mlp.down_proj"
    ).replace("feed_forward.w1", "mlp.gate_proj").replace("ffn_norm", "post_attention_layernorm").replace(
        "output.weight", "lm_head.weight"
    )


def remap_keys_qwen2(k: str) -> str:
    """Maps ZeroBand Qwen2 keys back to HuggingFace Qwen2 keys."""
    # Remove 'model.' prefix added during loading
    if k.startswith("model."):
        k = k[len("model.") :]

    # Handle lm_head separately (was adjusted during loading)
    if k == "lm_head.weight":
        return k  # Already correct HF key

    # Reverse the gate/up proj split
    m_gate = re.match(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)", k)
    if m_gate:
        layer_idx, wb = m_gate.groups()
        # We'll handle this when we see the corresponding up_proj
        return None  # Signal to skip this key for now

    m_up = re.match(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", k)
    if m_up:
        layer_idx, wb = m_up.groups()
        # Return the fused HF key name
        return f"layers.{layer_idx}.mlp.gate_up_proj.{wb}"

    # Default: should be a direct match after removing 'model.'
    return k


def _get_ffn_dim(hidden_dim: int, ffn_dim_multiplier: float, multiple_of: int) -> int:
    """Get the FFN dimension from ZeroBand args"""
    hidden_dim = int(8 * hidden_dim / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def convert_config_zb_to_hf(
    zb_config, with_debug_automap: bool = False, type_model: str = "llama3"
):
    """Convert ZeroBand config to HuggingFace config"""
    if type_model == "qwen2":
        config = Qwen2Config()
        config.hidden_size = zb_config.dim
        config.num_hidden_layers = zb_config.n_layers
        config.num_attention_heads = zb_config.n_heads
        config.num_key_value_heads = zb_config.n_kv_heads or zb_config.n_heads  # HF uses num_key_value_heads
        config.vocab_size = zb_config.vocab_size
        # Qwen2 uses intermediate_size directly if provided in ModelArgs
        config.intermediate_size = (
            zb_config.intermediate_size
            or _get_ffn_dim(zb_config.dim, zb_config.ffn_dim_multiplier, zb_config.multiple_of)
        )
        config.rms_norm_eps = zb_config.norm_eps
        config.rope_theta = float(zb_config.rope_theta)
        config.max_position_embeddings = (
            zb_config.max_position_embeddings or zb_config.max_seq_len
        )  # Use max_position_embeddings if available
        config.max_window_layers = zb_config.max_window_layers or zb_config.n_layers  # Default to all layers if not specified
        config.use_sliding_window = zb_config.use_sliding_window or False
        config.sliding_window = zb_config.sliding_window  # Can be None if use_sliding_window is False

        # Standard Qwen2 token IDs
        config.bos_token_id = 151643  # <|im_start|>
        config.eos_token_id = 151645  # <|im_end|>
        # config.pad_token_id = 151643 # Often set same as BOS or a dedicated pad token if vocab has one

        config.architectures = ["Qwen2ForCausalLM"]

        # Set torch_dtype based on export config
        config.torch_dtype = config.torch_dtype

        # Add other relevant fields if needed, e.g., tie_word_embeddings
        # config.tie_word_embeddings = False # Or True based on your model

    else:  # Default to Llama config
        config = LlamaConfig()
        config.hidden_size = zb_config.dim
        config.num_hidden_layers = zb_config.n_layers
        config.num_attention_heads = zb_config.n_heads
        config.num_key_value_heads = zb_config.n_kv_heads
        config.vocab_size = zb_config.vocab_size
        config.intermediate_size = _get_ffn_dim(zb_config.dim, zb_config.ffn_dim_multiplier, zb_config.multiple_of)
        config.rms_norm_eps = zb_config.norm_eps
        config.rope_theta = float(zb_config.rope_theta)
        config.max_position_embeddings = zb_config.max_seq_len

        if type_model == "llama2":
            config.bos_token_id = [1]
            config.eos_token_id = [2]
        else:
            config.bos_token_id = [128000]
            config.eos_token_id = [128001, 128008, 128009]

        config.architectures = ["LlamaForCausalLM"]

        # Rope scaling
        config.rope_scaling = {
            "original_max_position_embeddings": 8192,
            "rope_type": "default",
        }

        if with_debug_automap:
            config.auto_map = {
                "AutoConfig": "PrimeIntellect/prime-llama-debug--configuration_llama.LlamaConfig",
                "AutoModelForCausalLM": "PrimeIntellect/prime-llama-debug--modeling_llama.LlamaForCausalLM",
            }

    return config


@torch.no_grad
def convert_qk_from_complex_to_rotate_half(linear_weight: torch.FloatTensor, head_dim: int) -> torch.FloatTensor:
    """Converts the Q/K weight from complex to rotate half form.
    This is required because the rotary implementation in ZeroBand uses complex numbers which encodes even elements as real and odd number as complex.
    [0, 1, 2, 3] -> [0 + 1j, 2 + 3j]
    However, the HuggingFace implementation uses rotate_half which encodes top half as real and bottom half as complex.
    [0, 1, 2, 3] -> [0, 1] + [2, 3]j

    We thus need to permute the QK outputs to match the HuggingFace implementation.
    """
    new_weight = torch.zeros_like(linear_weight)

    num_heads = linear_weight.size(0) // head_dim
    hhd = head_dim // 2

    # This applies the riffle shuffle permutation to the outputs of the linear for each attn head
    # Even numbers go to the top half, odd numbers go to the bottom half
    for i in range(num_heads):
        new_weight[i * head_dim : (i * head_dim + hhd), :].copy_(
            linear_weight[i * head_dim + 0 : (i + 1) * head_dim : 2, :]
        )
        new_weight[i * head_dim + hhd : (i + 1) * head_dim, :].copy_(
            linear_weight[i * head_dim + 1 : (i + 1) * head_dim : 2, :]
        )

    return new_weight


@torch.no_grad()
def main(config: ExportConfig):
    logger = get_logger()
    logger.info(f"Exporting checkpoint from {config.ckpt.path} with config {config.model_dump_json(indent=2)}")

    # Resolve environment variables in paths
    resolve_env_vars(config)
    ckpt_path = Path(config.ckpt.path)
    resume_path = Path(config.ckpt.resume) if config.ckpt.resume else ckpt_path / "consolidated" / "consolidated.pth"
    save_path = Path(config.ckpt.save) if config.ckpt.save else ckpt_path / "hf_export"
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer from {config.data.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_path)

    logger.info(f"Loading ZeroBand model config for {config.type_model} {config.name_model}")
    if config.type_model == "qwen2":
        get_model_fn = get_model_qwen2
        ModelArgsClass = Qwen2ModelArgs
        remap_fn = remap_keys_qwen2
        convert_config_fn = convert_config_zb_to_hf_qwen2
        HFConfigClass = Qwen2Config
    elif config.type_model.startswith("llama"):
        get_model_fn = get_model_llama
        ModelArgsClass = LlamaModelArgs
        remap_fn = remap_keys_llama
        convert_config_fn = convert_config_zb_to_hf
        HFConfigClass = LlamaConfig
    else:
        raise ValueError(f"Unsupported model type for export: {config.type_model}")

    model, zb_model_args = get_model_fn(config, vocab_size=tokenizer.vocab_size)
    model = ModelWrapper(model)  # Wrap for checkpoint loading

    logger.info(f"Loading checkpoint state dict from {resume_path}")
    state_dict = torch.load(resume_path, map_location="cpu")
    if "model" in state_dict:  # Handle nested state dicts
        state_dict = state_dict["model"]

    # Filter out optimizer states if present
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("optim")}

    # Remap keys and handle fused layers for Qwen2
    hf_state_dict = {}
    pending_gate_projs = {}  # Store gate_proj tensors temporarily for Qwen2 fusion

    for k, v in state_dict.items():
        hf_key = remap_fn(k)
        if hf_key is None:  # Special handling for Qwen2 gate_proj
            if config.type_model == "qwen2":
                m_gate = re.match(r"model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)", k)
                if m_gate:
                    layer_idx, wb = m_gate.groups()
                    pending_gate_projs[(layer_idx, wb)] = v
                else:
                    logger.warning(f"Skipping unexpected key during Qwen2 remapping: {k}")
            continue  # Skip this key for now

        # Fuse gate_proj and up_proj for Qwen2
        if config.type_model == "qwen2":
            m_up = re.match(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)", hf_key)
            if m_up:
                layer_idx, wb = m_up.groups()
                if (layer_idx, wb) in pending_gate_projs:
                    gate_tensor = pending_gate_projs.pop((layer_idx, wb))
                    # Concatenate gate and up tensors
                    fused_tensor = torch.cat([gate_tensor, v], dim=0)
                    hf_state_dict[f"model.{hf_key}"] = fused_tensor.to(getattr(torch, config.torch_dtype))
                    continue  # Move to next key after fusion
                else:
                    logger.error(f"Found up_proj key {hf_key} but no matching gate_proj key was stored.")
                    # Fall through to default handling, might cause issues

        # Default handling for Llama and non-fused Qwen2 keys
        # Add 'model.' prefix if needed (HF usually expects it, except for lm_head)
        final_hf_key = hf_key if hf_key == "lm_head.weight" else f"model.{hf_key}"
        hf_state_dict[final_hf_key] = v.to(getattr(torch, config.torch_dtype))

    # Check if any gate_proj keys were left over for Qwen2
    if config.type_model == "qwen2" and pending_gate_projs:
        logger.error(f"Found unmatched gate_proj keys: {list(pending_gate_projs.keys())}")

    logger.info(f"Saving HuggingFace state dict with {len(hf_state_dict)} keys.")
    if config.save_format == "safetensors":
        save_fn = save_file
        save_name = "model.safetensors"
    else:
        save_fn = torch.save
        save_name = "pytorch_model.bin"

    save_fn(hf_state_dict, save_path / save_name)
    logger.info(f"Saved state dict to {save_path / save_name}")

    logger.info("Converting and saving HuggingFace config file.")
    hf_config = convert_config_fn(zb_model_args, config.with_debug_automap, type_model=config.type_model)
    hf_config.torch_dtype = config.torch_dtype  # Ensure dtype matches saved tensors
    hf_config.save_pretrained(save_path)
    logger.info(f"Saved config.json to {save_path}")

    logger.info("Saving tokenizer files.")
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved tokenizer files to {save_path}")

    # Save generation config (optional, basic example)
    gen_config = GenerationConfig(
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        # pad_token_id=hf_config.pad_token_id, # Uncomment if pad_token_id is set
    )
    gen_config.save_pretrained(save_path)
    logger.info(f"Saved generation_config.json to {save_path}")

    logger.info("Export complete.")


if __name__ == "__main__":
    logger = get_logger()
    config = ExportConfig(**parse_argv())
    resolve_env_vars(config)
    logger.debug(f"config: {config.model_dump()}")

    main(config)
