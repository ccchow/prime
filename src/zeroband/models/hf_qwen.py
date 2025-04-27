"""
Hugging Face Qwen model loader for Prime FSDP.

This module provides helper functions to load Qwen models and tokenizers from Hugging Face,
either from pretrained weights or from scratch based on configuration.
These models are specifically prepared for Prime's Fully Sharded Data Parallel (FSDP) training.
"""

import torch
from typing import Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from zeroband.config import Config


def load_qwen_model(config: Config) -> Tuple[torch.nn.Module, AutoConfig]:
    """Load a Hugging Face Qwen model for given config and prepare it for FSDP.

    - If config.hf_model_name is set, load that pretrained model.
    - Otherwise, raise an error (Qwen from scratch not supported).

    Args:
        config: The Prime FSDP configuration object

    Returns:
        Tuple containing:
            - model: Qwen model instance, ready for FSDP wrapping
            - model_config: Config instance used to create the model
    """
    if config.hf_model_name:
        model_name = config.hf_model_name
        model_config = AutoConfig.from_pretrained(model_name)
        # Update max position embeddings to match Prime config
        model_config.max_position_embeddings = config.data.seq_length
        # Load pretrained Qwen model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        raise ValueError("Loading Qwen model from scratch is not supported; please set config.hf_model_name.")

    # Wrap model to adapt Prime's interface
    model = QwenModelAdapter(base_model)

    # Expose transformer blocks for FSDP
    # Qwen models store layers in base_model.model.layers
    hf_blocks = base_model.model.layers  # List of transformer layers
    model.layers = {str(i): layer for i, layer in enumerate(hf_blocks)}

    return model, model_config


def load_qwen_tokenizer(config: Config) -> PreTrainedTokenizer:
    """Load a Hugging Face tokenizer for the specified Qwen model.

    Args:
        config: The Prime FSDP configuration object

    Returns:
        A properly configured PreTrainedTokenizer for the Qwen model
    """
    if config.data.fake and config.name_model == "debugmodel":
        raise ValueError("For fake data and debugmodel, use FakeTokenizer instead of this helper")

    if config.hf_model_name:
        tokenizer_name = config.hf_model_name
    else:
        raise ValueError("Loading Qwen tokenizer from scratch is not supported; please set config.hf_model_name.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=True)

    # Ensure pad token is set (important for proper batching)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    return tokenizer


class QwenModelAdapter(torch.nn.Module):
    """Adapter wrapper for Qwen models to match Prime's expected interface."""

    def __init__(self, model: AutoModelForCausalLM):
        super().__init__()
        self.model = model
        # Expose base model and output head
        self.transformer = getattr(model, 'model', model)
        self.output = getattr(model, 'lm_head', getattr(model, 'score', None))
        self.layers = None

    def forward(self, tokens, block_mask=None, **kwargs):
        """Adapt Prime's interface to Hugging Face Qwen's interface."""
        attention_mask = None
        if block_mask is not None:
            batch_size, seq_len = tokens.shape
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long,
                                        device=tokens.device)

        outputs = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            **kwargs
        )
        # Return logits
        return outputs.logits

    def set_requires_gradient_sync(self, requires_sync):
        """Support Prime's gradient synchronization control."""
        if hasattr(self.model, 'set_requires_gradient_sync'):
            self.model.set_requires_gradient_sync(requires_sync)


def load_qwen2_omni_model(config: Config) -> Tuple[torch.nn.Module, AutoConfig]:
    """
    Load the Qwen2.5-Omni-7B model from Hugging Face and wrap it for Prime FSDP.
    """
    model_name = config.hf_model_name or "Qwen/Qwen2.5-Omni-7B"
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.max_position_embeddings = config.data.seq_length
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = QwenModelAdapter(base_model)
    hf_blocks = base_model.model.layers
    model.layers = {str(i): layer for i, layer in enumerate(hf_blocks)}
    return model, model_config


def load_qwen2_omni_tokenizer(config: Config) -> PreTrainedTokenizer:
    """
    Load the Qwen2.5-Omni-7B tokenizer from Hugging Face.
    """
    tokenizer_name = config.hf_model_name or "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    return tokenizer
