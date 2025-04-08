"""
Hugging Face LLaMA model loader for Prime FSDP.

This module provides helper functions to load LLaMA models and tokenizers from Hugging Face,
either from pretrained weights or from scratch based on configuration.
These models are specifically prepared for Prime's Fully Sharded Data Parallel (FSDP) training approach.
"""

import torch
from typing import Tuple, Dict, Optional
from zeroband.config import Config
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, PreTrainedTokenizer


def load_llama_model(config: Config) -> Tuple[torch.nn.Module, LlamaConfig]:
    """Load a Hugging Face LLaMA model for given config and prepare it for FSDP.
    
    - If config.hf_model_name is set, load that pretrained model.
    - Otherwise, build a LLaMA model from scratch based on config.name_model size.
    
    The model is specifically prepared for Prime's FSDP approach:
    - Hugging Face transformer blocks are transformed into a dictionary format
      that Prime's FSDP wrapping loop expects.
    - This enables the same FSDP wrapping code to work with both Prime's custom
      models and HuggingFace models.
    
    Args:
        config: The Prime FSDP configuration object
        
    Returns:
        Tuple containing:
            - model: LlamaForCausalLM instance, ready for FSDP wrapping
            - model_config: LlamaConfig instance used to create the model
    
    Note: 
        This integration assumes Prime's use of the composable FSDP API 
        (torch.distributed._composable.fsdp.fully_shard) rather than the
        module-based wrapper approach.
    """
    # 1. Determine model configuration
    if config.hf_model_name is not None:
        # Use the specified Hugging Face model
        model_name = config.hf_model_name
        model_config = LlamaConfig.from_pretrained(model_name)
        
        # Update model config with any relevant values from Prime config
        model_config.max_position_embeddings = config.data.seq_length
    else:
        # Create a config based on model size from Prime's configuration
        if config.type_model == "llama2":
            # Map Prime model sizes to Hugging Face config parameters for Llama 2
            model_config = _create_llama2_config(config.name_model, config.data.seq_length)
        elif config.type_model == "llama3":
            # Map Prime model sizes to Hugging Face config parameters for Llama 3
            model_config = _create_llama3_config(config.name_model, config.data.seq_length)
        else:
            raise ValueError(f"Model type {config.type_model} not supported")
        
        # No specific HF model to load, but we'll initialize from config
        model_name = None
    
    # 2. Load or initialize LLaMA model
    if model_name is not None:
        # Load pretrained model from Hugging Face
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            config=model_config,
            torch_dtype=torch.bfloat16  # TODO: Using bfloat16 by default for better training stability
        )
    else:
        # Initialize a model from scratch with the config
        model = LlamaForCausalLM(model_config)
    
    # 3. Prepare model for FSDP wrapping
    # HuggingFace LLaMA models store transformer blocks in model.model.layers
    # Prime FSDP expects these layers in a dictionary at model.layers for iterative sharding
    hf_blocks = model.model.layers  # List of transformer layers
    
    # Convert to the format expected by Prime's FSDP wrapping loop
    # This ensures that the loop: for layer_id, transformer_block in model.layers.items(): 
    # works correctly with HuggingFace models
    model.layers = {str(i): layer for i, layer in enumerate(hf_blocks)}
    
    # Note: The embedding layer and LM head will be handled by the top-level 
    # fully_shard(model, ...) call in Prime's training script
    
    return model, model_config


def load_llama_tokenizer(config: Config) -> PreTrainedTokenizer:
    """Load a Hugging Face tokenizer for the specified LLaMA model.
    
    - If config.hf_model_name is set, load tokenizer from that model.
    - Otherwise, use default tokenizers based on the model type (llama2/llama3).
    
    Args:
        config: The Prime FSDP configuration object
        
    Returns:
        A properly configured PreTrainedTokenizer for the LLaMA model
    """
    # 1. Determine which tokenizer to load
    if config.data.fake and config.name_model == "debugmodel":
        # For fake data and debug model, we should use the FakeTokenizer
        # However, that's handled at a higher level, so we'll raise an error
        raise ValueError("For fake data and debugmodel, use FakeTokenizer instead of this helper")
    
    if config.hf_model_name is not None:
        # Load tokenizer from the specified Hugging Face model
        tokenizer_name = config.hf_model_name
    else:
        # Use default tokenizers based on model type
        # These match the current hardcoded values in Prime's train.py
        default_tokenizers = {
            "llama2": "mistralai/Mistral-7B-v0.1",  # Mistral tokenizer works well with LLaMA 2
            "llama3": "meta-llama/Meta-Llama-3-8B",  # Meta's official LLaMA 3 tokenizer
        }
        
        if config.type_model not in default_tokenizers:
            raise ValueError(f"Model type {config.type_model} not supported for tokenizer loading")
        
        tokenizer_name = default_tokenizers[config.type_model]
    
    # 2. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # 3. Set pad token if missing
    # Set pad_token for LLaMA if missing (important for proper batching)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "</s>"
    
    return tokenizer


def _create_llama2_config(name_model: str, seq_length: int) -> LlamaConfig:
    """Create a Hugging Face LlamaConfig for Llama 2 models based on size."""
    # Map Prime's model sizes to Hugging Face config parameters
    configs = {
        "debugmodel": {"hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 8},
        "70M": {"hidden_size": 512, "num_hidden_layers": 6, "num_attention_heads": 8},
        "150M": {"hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 16},
        "271M": {"hidden_size": 1024, "num_hidden_layers": 16, "num_attention_heads": 8},
        "1B": {"hidden_size": 2048, "num_hidden_layers": 18, "num_attention_heads": 16},
        "7B": {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32},
        "10B": {"hidden_size": 5120, "num_hidden_layers": 32, "num_attention_heads": 40},
        "13B": {"hidden_size": 5120, "num_hidden_layers": 40, "num_attention_heads": 40},
        "26B": {"hidden_size": 5120, "num_hidden_layers": 80, "num_attention_heads": 40},
        "70B": {
            "hidden_size": 8192, 
            "num_hidden_layers": 80, 
            "num_attention_heads": 64, 
            "num_key_value_heads": 8, 
            "intermediate_size": 28672
        },
    }
    
    if name_model not in configs:
        raise ValueError(f"Llama 2 model size '{name_model}' not supported")
    
    # Start with a base config and update with size-specific parameters
    config_params = {
        "vocab_size": 32000,  # Standard Llama vocabulary size
        "hidden_size": 4096,  # Will be overridden
        "intermediate_size": None,  # Will be computed if not explicitly specified
        "num_hidden_layers": 32,  # Will be overridden
        "num_attention_heads": 32,  # Will be overridden
        "num_key_value_heads": None,  # Group query attention if specified
        "hidden_act": "silu",
        "max_position_embeddings": seq_length,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "pad_token_id": None,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pretraining_tp": 1,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "architectures": ["LlamaForCausalLM"],
    }
    
    # Update with size-specific parameters
    config_params.update(configs[name_model])
    
    # Compute intermediate_size if not explicitly provided
    if config_params["intermediate_size"] is None:
        # Standard formula for Llama 2: (hidden_size * 8 / 3), rounded to nearest multiple of 256
        hidden_size = config_params["hidden_size"]
        intermediate_size = int((hidden_size * 8) / 3)
        # Round to nearest multiple of 256
        config_params["intermediate_size"] = 256 * ((intermediate_size + 255) // 256)
    
    return LlamaConfig(**config_params)


def _create_llama3_config(name_model: str, seq_length: int) -> LlamaConfig:
    """Create a Hugging Face LlamaConfig for Llama 3 models based on size."""
    # Map Prime's model sizes to Hugging Face config parameters
    configs = {
        "debugmodel": {"hidden_size": 256, "num_hidden_layers": 8, "num_attention_heads": 16},
        "1B": {
            "hidden_size": 2048, 
            "num_hidden_layers": 18, 
            "num_attention_heads": 16, 
            "num_key_value_heads": 8,
            "intermediate_size": None  # Will be computed with ffn_dim_multiplier=1.3
        },
        "8B": {
            "hidden_size": 4096, 
            "num_hidden_layers": 32, 
            "num_attention_heads": 32, 
            "num_key_value_heads": 8,
            "intermediate_size": None  # Will be computed with ffn_dim_multiplier=1.3
        },
        "10B": {
            "hidden_size": 4096, 
            "num_hidden_layers": 42, 
            "num_attention_heads": 32, 
            "num_key_value_heads": 8,
            "intermediate_size": None  # Will be computed with ffn_dim_multiplier=1.3
        },
        "70B": {
            "hidden_size": 8192, 
            "num_hidden_layers": 80, 
            "num_attention_heads": 64, 
            "num_key_value_heads": 8,
            "intermediate_size": None  # Will be computed with ffn_dim_multiplier=1.3
        },
        "405B": {
            "hidden_size": 16384, 
            "num_hidden_layers": 126, 
            "num_attention_heads": 128, 
            "num_key_value_heads": 8,
            "intermediate_size": None  # Will be computed with ffn_dim_multiplier=1.2
        },
    }
    
    if name_model not in configs:
        raise ValueError(f"Llama 3 model size '{name_model}' not supported")
    
    # Start with a base config and update with size-specific parameters
    config_params = {
        "vocab_size": 128256,  # Llama 3 vocabulary size
        "hidden_size": 4096,  # Will be overridden
        "intermediate_size": None,  # Will be computed
        "num_hidden_layers": 32,  # Will be overridden
        "num_attention_heads": 32,  # Will be overridden
        "num_key_value_heads": None,  # Group query attention
        "hidden_act": "silu",
        "max_position_embeddings": seq_length,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "pad_token_id": None,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pretraining_tp": 1,
        "tie_word_embeddings": False,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "original_max_position_embeddings": 8192,
            "rope_type": "default",
        },
        "architectures": ["LlamaForCausalLM"],
    }
    
    # Update with size-specific parameters
    config_params.update(configs[name_model])
    
    # Compute intermediate_size if not explicitly provided
    if config_params["intermediate_size"] is None:
        # Handle different model sizes with appropriate ffn_dim_multiplier
        hidden_size = config_params["hidden_size"]
        if name_model == "405B":
            ffn_dim_multiplier = 1.2
            multiple_of = 4096
        else:
            ffn_dim_multiplier = 1.3
            if hidden_size <= 2048:
                multiple_of = 512
            elif hidden_size <= 4096:
                multiple_of = 1024
            else:
                multiple_of = 4096
        
        # Compute using Llama 3 formula
        intermediate_size = int(((hidden_size * 2) / 3) * ffn_dim_multiplier)
        # Round to nearest multiple of the specified value
        config_params["intermediate_size"] = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
    
    return LlamaConfig(**config_params)