import torch
import pytest
from zeroband.models.qwen2.model import (
    ModelArgs,
    Qwen2Attention,
    Qwen2RotaryEmbedding,
    Qwen2ForCausalLM,
)


def test_attention_output_shape_dtype():
    # small config and input
    config = ModelArgs(dim=8, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=10, max_position_embeddings=4)
    batch, seq = 2, 4
    hidden_states = torch.randn(batch, seq, config.dim, dtype=torch.float32)
    # compute rotary embeddings
    pos_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    rotary = Qwen2RotaryEmbedding(config)
    cos, sin = rotary(hidden_states, pos_ids)
    # run attention
    attn = Qwen2Attention(config, layer_idx=0)
    output = attn(hidden_states, (cos, sin), attention_mask=torch.ones(batch, seq))
    # output without attentions returns Tensor
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch, seq, config.dim)
    assert output.dtype == hidden_states.dtype


def test_rotary_embeddings_deterministic_and_match_manual():
    config = ModelArgs(dim=8, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=10, max_position_embeddings=4)
    batch, seq = 2, 4
    x = torch.randn(batch, seq, config.dim)
    pos_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    rotary = Qwen2RotaryEmbedding(config)
    cos1, sin1 = rotary(x, pos_ids)
    cos2, sin2 = rotary(x, pos_ids)
    # deterministic
    assert torch.allclose(cos1, cos2)
    assert torch.allclose(sin1, sin2)
    # manual computation
    inv_freq, scaling = rotary.rope_init_fn(config, None)
    inv_freq_e = inv_freq[None, :, None].float().expand(batch, -1, 1)
    pos_e = pos_ids[:, None, :].float()
    freqs = (inv_freq_e @ pos_e).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_manual = emb.cos() * scaling
    sin_manual = emb.sin() * scaling
    assert torch.allclose(cos1, cos_manual.to(cos1.dtype))
    assert torch.allclose(sin1, sin_manual.to(sin1.dtype))


def test_causal_lm_forward_and_generation():
    config = ModelArgs(dim=8, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=20, max_position_embeddings=4)
    model = Qwen2ForCausalLM(config)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    # forward without labels
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    assert logits.shape == (1, 3, config.vocab_size)
    # generation: pick argmax of last token
    next_logits = logits[:, -1, :]
    next_token = torch.argmax(next_logits, dim=-1)
    assert next_token.shape == (1,)
