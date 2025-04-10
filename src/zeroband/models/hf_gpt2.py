"""
Hugging Face GPT-2 model loader for Prime FSDP.

This module provides helper functions to load GPT-2 models and tokenizers from Hugging Face,
either from pretrained weights or from scratch based on configuration.
These models are specifically prepared for Prime's Fully Sharded Data Parallel
(FSDP) training approach.
"""

from typing import Tuple

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizer

from zeroband.config import Config

def load_gpt2_model(config: Config) -> Tuple[torch.nn.Module, GPT2Config]:
    """Load a Hugging Face GPT-2 model based on Prime's configuration."""
    # Step 1: Determine GPT-2 configuration source
    if config.hf_model_name:
        # Load pretrained configuration from Hugging Face Hub or local path
        model_config = GPT2Config.from_pretrained(config.hf_model_name)

        # Update model config with any relevant values from Prime config
        model_config.n_positions = config.data.seq_length
    else:
        # Create a custom GPT-2 configuration from scratch based on model size
        model_config = _create_gpt2_config(config.name_model, config.data.seq_length)

    # Step 2: Instantiate GPT-2 model
    if config.hf_model_name:
        # Load GPT-2 model with pretrained weights
        base_model = GPT2LMHeadModel.from_pretrained(
            config.hf_model_name,
            config=model_config,
            torch_dtype=torch.bfloat16
        )
    else:
        # Instantiate GPT-2 model from scratch (randomly initialized weights)
        base_model = GPT2LMHeadModel(model_config)

    # Step 3: Wrap the model with our adapter for Prime compatibility
    model = GPT2ModelAdapter(base_model)

    # Step 4: Expose transformer blocks explicitly for Prime's FSDP integration
    gpt2_blocks = base_model.transformer.h
    model.layers = {str(i): layer for i, layer in enumerate(gpt2_blocks)}

    return model, model_config


class GPT2ModelAdapter(torch.nn.Module):
    """Adapter wrapper for GPT-2 models to match Prime's expected interface."""

    def __init__(self, model: GPT2LMHeadModel):
        super().__init__()
        self.model = model
        # Make important components directly accessible
        self.transformer = model.transformer
        self.output = model.lm_head  # Map output to match expected attribute
        self.layers = None

    def forward(self, tokens, block_mask=None, **kwargs):
        """Adapt Prime's interface to Hugging Face GPT-2's interface."""
        # Process block_mask for GPT-2 attention mask format
        attention_mask = None

        if block_mask is not None:
            batch_size, seq_len = tokens.shape
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long,
                                        device=tokens.device)

        # Forward pass with correctly shaped attention_mask
        outputs = self.model(
            input_ids=tokens,
            attention_mask=attention_mask,
            **kwargs
        )

        return outputs.logits

    def set_requires_gradient_sync(self, requires_sync):
        """Support Prime's gradient synchronization control."""
        if hasattr(self.model, 'set_requires_gradient_sync'):
            self.model.set_requires_gradient_sync(requires_sync)
        # If the base model doesn't have this method, it's a no-op


def load_gpt2_tokenizer(config: Config) -> PreTrainedTokenizer:
    """Load a Hugging Face tokenizer for the specified GPT-2 model.

    Args:
        config: The Prime FSDP configuration object

    Returns:
        A properly configured PreTrainedTokenizer for the GPT-2 model
    """
    if config.data.fake and config.name_model == "debugmodel":
        # For fake data and debug model, we should use the FakeTokenizer
        # However, that's handled at a higher level, so we'll raise an error
        raise ValueError("For fake data and debugmodel, use FakeTokenizer instead of this helper")

    # Determine which tokenizer to load
    if config.hf_model_name is not None:
        # Load tokenizer from the specified Hugging Face model
        tokenizer_name = config.hf_model_name
    else:
        # Default to standard GPT-2 tokenizer
        tokenizer_name = "gpt2"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Ensure pad token is set (important for proper batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _create_gpt2_config(name_model: str, seq_length: int) -> GPT2Config:
    """Create a GPT2Config based on Prime's model size naming.

    Maps Prime's model size names to appropriate GPT-2 configuration parameters.

    Args:
        name_model: Prime's model size name (e.g., "150M")
        seq_length: Maximum sequence length for the model
    
    Returns:
        GPT2Config with appropriate parameters for the requested model size
    """
    # Map Prime's model sizes to GPT-2 config parameters
    # These configurations approximate standard GPT-2 sizes while aligning with Prime naming
    configs = {
        "debugmodel": {"n_embd": 256, "n_layer": 2, "n_head": 8},
        "70M": {"n_embd": 512, "n_layer": 6, "n_head": 8},
        "150M": {"n_embd": 768, "n_layer": 12, "n_head": 12},  # Similar to standard GPT-2
        "271M": {"n_embd": 1024, "n_layer": 16, "n_head": 16},
        "1B": {"n_embd": 1536, "n_layer": 20, "n_head": 24},
        "7B": {"n_embd": 2048, "n_layer": 32, "n_head": 32},  # Larger than standard GPT-2 sizes
        "10B": {"n_embd": 2560, "n_layer": 36, "n_head": 40},
    }

    if name_model not in configs:
        raise ValueError(f"GPT-2 model size '{name_model}' not supported")

    # Start with base config and update with size-specific parameters
    config_params = {
        "vocab_size": 50257,  # Standard GPT-2 vocabulary size
        "n_positions": seq_length,
        "embd_pdrop": 0.1,
        "attn_pdrop": 0.1,
        "resid_pdrop": 0.1,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "architectures": ["GPT2LMHeadModel"],
    }

    # Update with size-specific parameters
    config_params.update(configs[name_model])

    return GPT2Config(**config_params)
