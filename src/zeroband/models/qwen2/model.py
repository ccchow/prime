# Custom Qwen2 implementation for Prime, based on HuggingFace Qwen2 code and zeroband Llama model style.

import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple, List
from zeroband.models.norms import build_norm
from zeroband.config import AttnFnType
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward, ALL_ATTENTION_FUNCTIONS

# TODO: import or implement rotary embedding utilities from HF.

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make hidden size multiple of this
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    # TODO: rope_scaling parameters
    max_position_embeddings: int = 32768
    use_sliding_window: bool = False
    sliding_window: Optional[int] = None
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    attn_impl: str = "eager"  # or 'flash', 'sdpa', etc.

# TODO: implement bytes_to_unicode, get_pairs, BPE encoder if needed or use HF tokenizer externally.

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelArgs, device=None):
        super().__init__()
        # Determine rope type
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        # initialize inverse frequency and scaling
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

    def _dynamic_frequency_update(self, position_ids: torch.LongTensor, device: torch.device):
        seq_len = int(torch.max(position_ids)) + 1
        # grow cache
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        # reset to original
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # dynamic update if needed
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, x.device)
        # compute frequencies
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # determine autocast device
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen2Attention(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.head_dim = config.dim // self.n_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.shape
        # project to multi-head QKV
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # apply rotary
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # update kv cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
        # sliding window
        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        # choose attention implementation
        if self.config.attn_impl == "eager":
            attention_fn = eager_attention_forward
        else:
            attention_fn = ALL_ATTENTION_FUNCTIONS[self.config.attn_impl]
        # compute attention
        attn_output, attn_weights = attention_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            sliding_window=sliding_window,
        )
        # reshape and output projection
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights) if output_attentions else attn_output

class Qwen2MLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        # TODO: ensure ModelArgs has `intermediate_size` matching Qwen2Config
        intermediate_size = config.intermediate_size  # placeholder; add to ModelArgs
        self.gate_proj = nn.Linear(config.dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.dim, bias=False)
        self.act_fn = F.silu  # using SiLU as Qwen default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gated activation: down_proj(act(gate_proj(x)) * up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        # TODO: instantiate attention, mlp, input and post-attn norms
        raise NotImplementedError("Qwen2DecoderLayer init not implemented")

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None):
        # TODO: implement decoder layer forward pass
        raise NotImplementedError("Qwen2DecoderLayer forward not implemented")

class Qwen2Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        # TODO: implement embed_tokens, layers, rotary_emb, final norm
        raise NotImplementedError("Qwen2Model init not implemented")

    def forward(self, input_ids: torch.LongTensor = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None, **kwargs):
        # TODO: implement forward pass
        raise NotImplementedError("Qwen2Model forward not implemented")

class Qwen2ForCausalLM(nn.Module):  # TODO: extend GenerationMixin if available
    def __init__(self, config: ModelArgs):
        super().__init__()
        # TODO: instantiate Qwen2Model and lm_head
        raise NotImplementedError("Qwen2ForCausalLM init not implemented")

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, cache_position=None, num_logits_to_keep=0, **kwargs):
        # TODO: implement forward logic, compute logits and loss
        raise NotImplementedError("Qwen2ForCausalLM forward not implemented")
