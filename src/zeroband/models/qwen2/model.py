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
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from transformers.modeling_outputs import CausalLMOutputWithPast
import math

# TODO: import or implement rotary embedding utilities from HF.

@dataclass
class ModelArgs:
    hidden_size: int = 4096
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    num_attention_heads: int = 32
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
    intermediate_size: Optional[int] = None     # new — FFN hidden size

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
        # Number of repeats of each KV head to match query heads
        self.num_key_value_groups = self.n_heads // self.n_kv_heads
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
        block_mask: Optional[BlockMask] = None,
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
        # if block_mask provided, use block-sparse flex attention
        if block_mask is not None:
            attn_output = flex_attention(q, k, v, block_mask=block_mask)
            attn_weights = None
        else:
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
        # derive intermediate size if not explicitly set
        if config.intermediate_size is None:
            mult = config.ffn_dim_multiplier or 4
            inter = int(math.ceil((config.dim * mult) / config.multiple_of) * config.multiple_of)
        else:
            inter = config.intermediate_size
        self.gate_proj = nn.Linear(config.dim, inter, bias=False)
        self.up_proj   = nn.Linear(config.dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.dim, bias=False)
        self.act_fn = F.silu  # using SiLU as Qwen default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gated activation: down_proj(act(gate_proj(x)) * up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        # layer normalization before attention
        self.input_layernorm = build_norm("rmsnorm", config.dim, config.norm_eps)
        # self-attention module
        self.self_attn = Qwen2Attention(config, layer_idx)
        # layer normalization before MLP
        self.post_attention_layernorm = build_norm("rmsnorm", config.dim, config.norm_eps)
        # gated feed-forward module
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        block_mask: Optional[BlockMask] = None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ):
        # Self-attention block
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            normed,
            position_embeddings,
            attention_mask,
            past_key_value,
            cache_position,
            block_mask=block_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        if output_attentions:
            attn_output, attn_weights = attn_out
        else:
            attn_output = attn_out
            attn_weights = None
        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed)
        hidden_states = residual + mlp_output

        # prepare outputs
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

class Qwen2Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        # token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)  # TODO: allow custom pad_token_id
        # rotary positional embedding
        self.rotary_emb = Qwen2RotaryEmbedding(config)
        # transformer layers
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config, i) for i in range(config.n_layers)])
        # final normalization
        self.norm = build_norm("rmsnorm", config.dim, config.norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        block_mask: Optional[BlockMask] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
        cache_position=None,
        **kwargs
    ):
        # input embeddings
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        batch_size, seq_len, _ = hidden_states.size()
        # position ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        # rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        # build causal attention mask
        if attention_mask is not None:
            # 2D mask to float mask with -inf for masked positions
            causal = torch.tril(torch.ones((seq_len, seq_len), device=hidden_states.device))
            attn_mask = attention_mask.view(batch_size,1,1,seq_len) * causal.unsqueeze(0)
            attn_mask = (1.0 - attn_mask) * -1e9
        else:
            attn_mask = None
        # iterate layers
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # forward layer
            outputs = layer(
                hidden_states,
                attention_mask=attn_mask,
                block_mask=block_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
            )
            # unpack
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions += (outputs[1],)
        # final norm
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        # return tuple
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs += (all_hidden_states,)
        if output_attentions:
            outputs += (all_attentions,)
        return outputs

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        # base model
        self.model = Qwen2Model(config)
        # language modeling head
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # tie weights
        try:
            self.lm_head.weight = self.model.embed_tokens.weight
        except Exception:
            pass
        # no direct module exposure; use properties for consistency

    @property
    def layers(self):
        # Return a dict of layer index to decoder layer for FSDP sharding
        return dict(enumerate(self.model.layers))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        block_mask: Optional[BlockMask] = None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False,
        cache_position=None,
        num_logits_to_keep: int = 0,
        **kwargs
    ):
        # run base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            block_mask=block_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        # compute logits
        logits = self.lm_head(hidden_states)
        # slice logits if needed
        if num_logits_to_keep and num_logits_to_keep > 0:
            logits = logits[..., :num_logits_to_keep]
        loss = None
        if labels is not None:
            # shift so that tokens < n predict n
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
            )
        if return_dict:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=outputs[1] if output_hidden_states else None,
                attentions=outputs[2] if output_attentions else None,
            )
        # else return tuple
        result = ()
        if loss is not None:
            result += (loss,)
        result += (logits,)
        if output_hidden_states:
            result += (outputs[1],)
        if output_attentions:
            result += (outputs[2],)
        return result
