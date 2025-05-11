from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# We assume a QwenTokenizer (or similar) will be used externally for text tokenization.
# The vocabulary includes text tokens and special multimodal tokens (e.g. <image>, <audio_start>, <audio_end>, etc.).
# For audio output tokens, we assume a discrete tokenizer (e.g. qwen-tts-tokenizer) provides a codebook of a fixed size.

@dataclass
class QwenModelArgs:
    """
    Configuration for Qwen2.5-Omni Thinker (main LLM) and Talker (audio decoder).
    """
    # Thinker (LLM) configuration
    dim: int = 3584                 # Hidden dimension of the transformer
    n_layers: int = 28             # Number of transformer layers
    n_heads: int = 28              # Number of attention heads (for queries)
    n_kv_heads: int = 4            # Number of key/value heads (for multi-query attention)
    vocab_size: int = 152064       # Text vocabulary size (includes special tokens)
    multiple_of: int = 256         # Feedforward hidden size is multiple of this
    ffn_dim_multiplier: float = None  # Optional custom multiplier for FFN dim
    norm_eps: float = 1e-5         # Epsilon for layer norm/RMS norm
    rope_theta: float = 1e6        # Base rotary position embedding theta (large for long context)
    max_seq_len: int = 32768       # Max sequence length for positional embeddings
    norm_type: str = "fused_rmsnorm" # Layer norm type
    depth_init: bool = True        # Depth-scaled initialization
    
    # Talker (audio AR decoder) configuration
    enable_talker: bool = True     # Whether to enable audio output generation
    audio_codebook_size: int = 128 # Number of possible audio tokens (codebook size)
    talker_n_layers: int = 24      # Number of layers in talker model
    talker_n_heads: int = 12       # Number of attention heads in talker
    talker_n_kv_heads: int = 4     # Number of KV heads in talker (for multi-query)
    talker_dim: int = 896          # Hidden dimension in talker
    # We use the main model dim for encoder context and token embeddings for talker:
    enc_dim: int = 3584            # Dimension of encoder (thinker) outputs and talker input embeddings
    talker_norm_eps: float = 1e-5  # Epsilon for talker norm

def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    """
    Precompute complex rotary embeddings for given dimension and sequence length.
    Uses base `theta` for frequency scaling (e.g. 1e6 for extended context)&#8203;:contentReference[oaicite:6]{index=6}.
    """
    # Compute geometric progression of frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs)  # (end, dim/2)
    # Convert to complex representation
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 tensor

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple:
    """
    Apply rotary position embedding to query and key tensors.
    `freqs_cis` is a tensor of shape [seq_len, dim/2] in complex form.
    Splits the last dimension of x into pairs and applies rotation.
    """
    # Assuming x shape: (batch, seqlen, n_heads, head_dim)
    # Split head_dim into half for complex rotation
    head_dim = xq.size(-1)
    half = head_dim // 2
    if half * 2 != head_dim:
        return xq, xk  # if head_dim is not even, skip RoPE
    # Convert to complex by view as two halves
    xq_half = xq[..., :half].float() + 1j * xq[..., half:].float()
    xk_half = xk[..., :half].float() + 1j * xk[..., half:].float()
    # Multiply by precomputed phase for positions
    freqs = freqs_cis[:xq.size(1)]  # (seqlen, half)
    # Broadcast freqs to batch, head dims and multiply
    xq_half = xq_half * freqs
    xk_half = xk_half * freqs
    # Convert back to real
    xq_rotated = torch.cat([torch.real(xq_half), torch.imag(xq_half)], dim=-1)
    xk_rotated = torch.cat([torch.real(xk_half), torch.imag(xk_half)], dim=-1)
    return xq_rotated.type_as(xq), xk_rotated.type_as(xk)

def build_norm(norm_type: str, dim: int, eps: float):
    """
    Build a normalization layer. Supports 'fused_rmsnorm' or 'layernorm' types.
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    # Default to RMSNorm (no bias) – fused if available
    try:
        from apex.normalization import FusedRMSNorm
        return FusedRMSNorm(dim, eps=eps)
    except ImportError:
        # Fallback: simple RMSNorm implementation
        class RMSNorm(nn.Module):
            def __init__(self, normalized_shape, eps=1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(normalized_shape))
            def forward(self, x: torch.Tensor):
                # RMSNorm: scale by L2 norm of features
                norm_x = x.norm(2, dim=-1, keepdim=True)
                return x / (norm_x + self.eps) * self.weight
        return RMSNorm(dim, eps=eps)

class Attention(nn.Module):
    """
    Multi-head self-attention module (supports multi-query attention).
    Projects input to Q, K, V and computes scaled dot-product attention.
    """
    def __init__(self, model_args: QwenModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads if model_args.n_kv_heads is not None else model_args.n_heads
        self.head_dim = model_args.dim // model_args.n_heads
        # Linear layers for query, key, value, and output
        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)
        self.attn_fn = getattr(model_args, "attn_fn", "flex")  # 'flex' for compiled or 'math'
    
    def init_weights(self, init_std: float):
        # Initialize Q, K, V with small std (truncated normal), and output with init_std
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
    
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, block_mask=None):
        # x: (batch, seq_len, dim)
        B, S, _ = x.size()
        # Project to queries, keys, values
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)
        # Apply rotary embeddings to q and k
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        # If using multi-query, repeat k, v to match number of query heads
        if self.n_kv_heads != self.n_heads:
            # Repeat keys/values n_rep times (n_heads//n_kv_heads)
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=2)
            v = v.repeat_interleave(n_rep, dim=2)
        # Transpose for attention computation: (B, n_heads, S, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Compute scaled dot-product attention (with causal mask)
        if block_mask is None:
            # Use PyTorch's optimized attention (SDPA) if available
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else:
            # If a BlockMask for sequences is provided (custom attention windowing), use it
            # Here we assume a compiled function exists; otherwise default to SDPA
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        # attn_out: (B, n_heads, S, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)  # (B, S, dim)
        return self.wo(attn_out)

class FeedForward(nn.Module):
    """
    Feed-forward network with gated activation (SwiGLU).
    Implements the 3 linear layer pattern: w1 * sigmoid(w3) -> w2.
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: float = None):
        super().__init__()
        # Compute hidden dimension (with 2/3 factor if gating is used)
        hidden_dim = int(2 * hidden_dim / 3)  # 2/3 of 4*dim = 8/3 * dim by default
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        # Round hidden_dim to nearest multiple
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # Linear layers: two "in->hidden" (w1, w3) and one "hidden->out" (w2)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation: silu(w1(x)) * w3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    def init_weights(self, init_std: float):
        # w1 with standard 0.02, w2 and w3 with layer-specific init_std
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3.weight, mean=0.0, std=init_std)

class TransformerBlock(nn.Module):
    """
    A single Transformer block consisting of self-attention and feed-forward sublayers.
    """
    def __init__(self, layer_id: int, model_args: QwenModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_layers = model_args.n_layers
        # Attention and FeedForward submodules
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=4 * model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier
        )
        # Layer norms (RMSNorm) for attention and FFN
        self.attention_norm = build_norm(model_args.norm_type, model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = build_norm(model_args.norm_type, model_args.dim, eps=model_args.norm_eps)
        # Determine init scaling for this layer
        if model_args.depth_init:
            # Depth-scaled initialization: std = 0.02 / sqrt(2*(layer_id+1))
            self.weight_init_std = 0.02 / ((2 * (layer_id + 1)) ** 0.5)
        else:
            # Uniform scaling across layers: std = 0.02 / sqrt(2*n_layers)
            self.weight_init_std = 0.02 / ((2 * self.n_layers) ** 0.5)
    def init_weights(self):
        # Initialize submodules using computed std
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)
        # Norm layers reset (RMSNorm weight to 1, already done in build_norm)
        if hasattr(self.attention_norm, 'reset_parameters'):
            self.attention_norm.reset_parameters()
        if hasattr(self.ffn_norm, 'reset_parameters'):
            self.ffn_norm.reset_parameters()
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, block_mask=None) -> torch.Tensor:
        # Apply self-attention
        h = x + self.attention(self.attention_norm(x), freqs_cis, block_mask=block_mask)
        # Apply feed-forward
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class QwenTransformer(nn.Module):
    """
    Qwen2.5-Omni Thinker Transformer Model (backbone LLM).
    Optionally integrates multimodal inputs (image, audio) and an audio Talker for speech output.
    """
    def __init__(self, model_args: QwenModelArgs):
        super().__init__()
        self.model_args = model_args
        # Token embedding for text (and possibly all tokens in the unified vocab)
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        # Rotary positional encodings (precomputed up to 2*max_seq_len for safety&#8203;:contentReference[oaicite:30]{index=30})
        freq_cis = precompute_freqs_cis(model_args.dim // model_args.n_heads, model_args.max_seq_len * 2, model_args.rope_theta)
        # Register the precomputed RoPE frequencies as a persistent buffer
        self.register_buffer("freqs_cis", freq_cis, persistent=True)
        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(i, model_args) for i in range(model_args.n_layers)])
        # Final RMSNorm (or LayerNorm) before output
        self.norm = build_norm(model_args.norm_type, model_args.dim, eps=model_args.norm_eps)
        # LM output head
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        # Optional Talker sub-model for audio token generation
        self.enable_talker = model_args.enable_talker
        if self.enable_talker:
            self._init_talker(model_args)
        # Initialize weights
        self.init_weights()
    
    def _init_talker(self, model_args: QwenModelArgs):
        """Initialize the Talker submodule for audio output (cross-modal decoder)."""
        # Project encoder (thinker) outputs to talker key/value dimension if needed
        enc_dim = model_args.dim  # 3584
        dec_dim = model_args.talker_dim  # 896
        # If encoder and decoder dims differ, use a linear projection for cross-attention keys/values
        if enc_dim != dec_dim:
            self.encoder_proj = nn.Linear(enc_dim, dec_dim, bias=False)
        else:
            self.encoder_proj = nn.Identity()
        # Token embedding for audio tokens (codebook of size audio_codebook_size)
        self.audio_embeddings = nn.Embedding(model_args.audio_codebook_size, model_args.enc_dim)
        # Talker decoder transformer layers
        self.talker_layers = nn.ModuleList([
            # Each talker layer has self-attn + cross-attn + ffn. We reuse a similar structure but must add cross-attn.
            # We'll implement cross-attn in forward (on the fly) for simplicity.
            TransformerBlock(i, QwenModelArgs(
                dim=model_args.talker_dim, n_layers=model_args.talker_n_layers, n_heads=model_args.talker_n_heads,
                n_kv_heads=model_args.talker_n_kv_heads, multiple_of=model_args.multiple_of, 
                ffn_dim_multiplier=model_args.ffn_dim_multiplier, norm_eps=model_args.talker_norm_eps,
                norm_type=model_args.norm_type, depth_init=model_args.depth_init, rope_theta=model_args.rope_theta,
                max_seq_len= model_args.max_seq_len  # talker may generate sequences (audio) up to similar lengths (or use sliding windows)
            ))
            for i in range(model_args.talker_n_layers)
        ])
        # Norm and output head for talker
        self.talker_norm = build_norm(model_args.norm_type, model_args.talker_dim, eps=model_args.talker_norm_eps)
        self.audio_output = nn.Linear(model_args.talker_dim, model_args.audio_codebook_size, bias=False)
    
    def init_weights(self):
        # Initialize token embeddings and output
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0)  # standard normal for embeddings
        for layer in self.layers:
            layer.init_weights()
        if self.enable_talker:
            # Initialize talker components
            if isinstance(self.encoder_proj, nn.Linear):
                nn.init.xavier_normal_(self.encoder_proj.weight)
            nn.init.normal_(self.audio_embeddings.weight, mean=0.0, std=1.0)
            for layer in self.talker_layers:
                layer.init_weights()
            nn.init.trunc_normal_(self.audio_output.weight, mean=0.0, std=(self.model_args.talker_dim ** -0.5))
    
    def forward(self, input_ids: torch.Tensor, attention_mask=None, images=None, audio=None):
        """
        Forward pass for text (and multimodal) inputs.
        input_ids: [batch, seq_len] text token ids (with special tokens for image/audio placeholders).
        images: optional precomputed image embeddings (list or tensor) aligned to placeholders.
        audio: optional audio input features (for ASR), to be integrated similarly.
        Returns:
          logits: [batch, seq_len, vocab_size] for next token prediction (text logits).
        """
        B, S = input_ids.shape
        # Embed tokens
        x = self.tok_embeddings(input_ids)  # [B, S, dim]
        # If image features are provided, insert them at the positions of image placeholder tokens
        # (In practice, processor would handle replacing <image> token with a single special id; here we assume one vector per image token)
        if images is not None:
            # images: list or tensor of shape [B, N_img, enc_dim]
            # We assume input_ids contains a special token (e.g. id == image_token_index) for each image.
            # Replace those token embeddings with provided image feature vectors.
            img_embed = images if isinstance(images, torch.Tensor) else torch.tensor(images, device=x.device, dtype=x.dtype)
            # Expand or index into img_embed for each placeholder occurrence
            # (For simplicity, assume one image token per sequence and use the first image embedding)
            # In a general implementation, we'd map each occurrence.
            x[input_ids == self.model_args.__dict__.get('image_token_index', 0)] = img_embed.to(x.device)
        # (Audio input integration could be handled similarly if needed for speech recognition.)
        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, self.freqs_cis, block_mask=None)
        # Final normalization
        x = self.norm(x)
        # Language model logits for text tokens
        logits = self.output(x)
        return logits if not self.enable_talker else (logits, x)

    def generate(self, input_ids, max_new_tokens=50, use_audio_in_video=False):
        """
        Generate text (and audio if applicable) given input prompt.
        Returns text_token_ids and optionally audio_waveform (if talker is enabled).
        """
        # Move model to eval mode
        self.eval()
        device = input_ids.device
        # 1. Text generation (autoregressive)
        generated_ids = [tok.item() for tok in input_ids[0]]  # assume single batch for simplicity
        # Determine special tokens
        audio_start_id = getattr(self.model_args, 'audio_start_token_id', None) or self.model_args.__dict__.get('audio_start_token_id')
        audio_end_id = getattr(self.model_args, 'audio_end_token_id', None) or self.model_args.__dict__.get('audio_end_token_id')
        for _ in range(max_new_tokens):
            inp = torch.tensor([generated_ids], dtype=torch.long, device=device)
            logits = self.forward(inp)[0] if self.enable_talker else self.forward(inp)
            next_token_logits = logits[:, -1, :]  # last token logits
            next_token_id = int(torch.argmax(next_token_logits, dim=-1))
            generated_ids.append(next_token_id)
            # Stop if end of sequence (audio or text)
            if next_token_id == audio_start_id or next_token_id == self.model_args.__dict__.get('eos_token_id'):
                break
        text_ids = generated_ids
        # 2. Audio generation using talker (if enabled and audio_start token was produced)
        audio_wave = None
        if self.enable_talker and audio_start_id is not None and audio_start_id in generated_ids:
            # Everything after the <audio_start> token will be generated by talker
            # Prepare encoder (thinker) hidden states as memory for talker
            seq = torch.tensor([generated_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                # Get encoder hidden states by forwarding through backbone (excluding output layer)
                x = self.tok_embeddings(seq)
                for layer in self.layers:
                    x = layer(x, self.freqs_cis)
                memory = self.norm(x)  # [1, seq_len, dim]
            memory = self.encoder_proj(memory)  # project to talker dim if needed
            # AR generation of audio tokens
            audio_tokens = []
            prev_token = torch.tensor([[0]], device=device)  # start from dummy (will replace with <audio_start>)
            # We use <audio_start> as first input to talker
            if audio_start_id is not None:
                prev_token = torch.tensor([[0]], device=device)  # index 0 as dummy; we will use embedding of audio_start
            prev_embed = self.audio_embeddings.weight[audio_tokens[0] if audio_tokens else 0] if audio_tokens else self.audio_embeddings.weight[0]
            # Actually use audio_start embedding as first input
            prev_embed = self.audio_embeddings.weight[0]  # assuming index 0 corresponds to audio start token embedding in audio_embeddings
            # Generate until audio_end token
            for t in range(1000):  # limit audio token generation length
                # One-step of talker: self-attention + cross-attention using memory
                # Using a simple loop to simulate decoding: not optimized
                if t == 0:
                    # initialize talker hidden state with audio_start
                    dec_h = prev_embed.view(1, 1, -1)  # shape [1,1,enc_dim]
                    # Project to talker hidden dim if needed
                    if self.model_args.enc_dim != self.model_args.talker_dim:
                        dec_h = self.encoder_proj(dec_h)  # reuse encoder_proj for embedding dimension adjustment
                else:
                    dec_h = torch.cat([dec_h, h_t], dim=1)  # append new token hidden state
                # Self-attention in talker (last token attends to previous tokens)
                for layer in self.talker_layers:
                    # We simulate a decoder with causal self-attn
                    dec_h = layer(dec_h, self.freqs_cis[:dec_h.size(1)], block_mask=None)
                    # Here, we would integrate cross-attention with encoder `memory` within each layer.
                    # For simplicity, we skip explicit cross-attn calculation and assume talker layers attend to `memory` (the thinker's hidden).
                    # In a full implementation, we would do something like:
                    # cross_attn_out = cross_attention(dec_h_last, memory)
                    # dec_h = dec_h + cross_attn_out
                    # (with appropriate normalization).
                    pass
                dec_out = self.talker_norm(dec_h)
                logits_audio = self.audio_output(dec_out)  # [1, seq_len, codebook_size]
                next_code = int(torch.argmax(logits_audio[:, -1, :], dim=-1))
                audio_tokens.append(next_code)
                if next_code == (audio_end_id or -1):
                    break
                # Prepare next input embedding
                h_t = self.audio_embeddings(torch.tensor([[next_code]], device=device))
            # Convert audio_tokens to waveform using vocoder (BigVGAN)
            # (In this pseudo-code, we'll skip actual vocoder implementation.)
            audio_wave = None  # Placeholder: would call BigVGAN or similar to decode tokens
        return text_ids, audio_wave
