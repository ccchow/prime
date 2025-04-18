# Ported Qwen2.5-Omni-7B architecture to Prime custom model implementation.
# Adapted from HuggingFace: https://huggingface.co/Qwen/Qwen2.5-Omni-7B

import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from zeroband.models.norms import build_norm
from zeroband.config import AttnFnType

from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE
from torch.nn.attention import SDPBackend, sdpa_kernel

_flex_attention_compiled = torch.compile(flex_attention, dynamic=False)

@torch.compiler.disable(recursive=False)
def flex_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: BlockMask,
) -> torch.Tensor:
    return _flex_attention_compiled(q, k, v, block_mask=block_mask)

from zeroband.models.llama.model import (
    precompute_freqs_cis,
    reshape_for_broadcast,
    apply_rotary_emb,
    repeat_kv,
    seqlens_to_docs_tensor,
    create_block_mask_from_seqlens,
    Attention as BaseAttention,
    FeedForward as BaseFeedForward,
    TransformerBlock as BaseTransformerBlock,
    Transformer as BaseTransformer,
)

dataclass
class QwenModelArgs:
    dim: int = 3584
    n_layers: int = 28
    n_heads: int = 28
    n_kv_heads: Optional[int] = 4
    vocab_size: int = 152064
    intermediate_size: int = 18944
    multiple_of: int = 256
    norm_eps: float = 1e-6
    rope_theta: float = 1e6
    max_seq_len: int = 32768
    depth_init: bool = True
    norm_type: str = "rmsnorm"
    attn_fn: AttnFnType = "flex"
    hidden_act: str = "silu"
    use_cache: bool = True

class QwenAttention(BaseAttention):
    """
    Qwen-specific attention using BaseAttention implementation.
    """
    def __init__(self, args: QwenModelArgs):
        super().__init__(args)

class QwenFeedForward(BaseFeedForward):
    """
    Qwen feedforward network, uses intermediate_size as hidden dimension.
    """
    def __init__(self, args: QwenModelArgs):
        super().__init__(
            dim=args.dim,
            hidden_dim=args.intermediate_size,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=None,
        )

class QwenTransformerBlock(BaseTransformerBlock):
    """
    Transformer block for Qwen, inherits BaseTransformerBlock.
    """
    def __init__(self, layer_id: int, args: QwenModelArgs):
        super().__init__(layer_id, args)

class QwenTransformer(BaseTransformer):
    """
    Qwen Transformer model.
    """
    def __init__(self, args: QwenModelArgs):
        super().__init__(args)

    def _precompute_freqs_cis(self) -> torch.Tensor:
        return precompute_freqs_cis(
            self.model_args.dim // self.model_args.n_heads,
            self.model_args.max_seq_len * 2,
            self.model_args.rope_theta,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "QwenTransformer":
        """
        Load weights from HuggingFace Qwen checkpoint and map to Prime model.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        # load HF config and override args
        hf_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config_kwargs = {k: getattr(hf_config, k) for k in hf_config.__dict__ if k in QwenModelArgs.__annotations__}
        config_kwargs.update(kwargs)
        args = QwenModelArgs(**config_kwargs)

        # initialize prime model
        model = cls(args)

        # load HF model state
        hf_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
        hf_state = hf_model.state_dict()

        # remap HF keys to Prime keys (simple passthrough, customize as needed)
        mapped_state = {}
        for hf_key, hf_val in hf_state.items():
            prime_key = hf_key
            # TODO: insert key mapping rules here
            mapped_state[prime_key] = hf_val

        model.load_state_dict(mapped_state, strict=False)
        return model
