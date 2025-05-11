# filepath: src/zeroband/models/qwen/__init__.py
"""
Prime custom Qwen model support: expose get_model for training.
"""
from dataclasses import dataclass
from zeroband.config import Config
from .model import QwenTransformer, QwenModelArgs

__all__ = ["get_model"]

def get_model(
    config: Config,
    vocab_size: int,
) -> tuple[QwenTransformer, QwenModelArgs]:
    """
    Instantiate custom Qwen Transformer with Prime's model args.
    """
    # prepare args from config
    args = QwenModelArgs(
        dim=config.model_args.dim if hasattr(config, 'model_args') else QwenModelArgs.dim,
        n_layers=config.model_args.n_layers if hasattr(config, 'model_args') else QwenModelArgs.n_layers,
        n_heads=config.model_args.n_heads if hasattr(config, 'model_args') else QwenModelArgs.n_heads,
        n_kv_heads=config.model_args.n_kv_heads if hasattr(config, 'model_args') else QwenModelArgs.n_kv_heads,
        vocab_size=vocab_size,
        intermediate_size=config.model_args.intermediate_size if hasattr(config, 'model_args') else QwenModelArgs.intermediate_size,
        multiple_of=QwenModelArgs.multiple_of,
        norm_eps=QwenModelArgs.norm_eps,
        rope_theta=QwenModelArgs.rope_theta,
        max_seq_len=config.data.seq_length,
        depth_init=True,
        norm_type=QwenModelArgs.norm_type,
        attn_fn=config.train.attn_fn,
        hidden_act=QwenModelArgs.hidden_act,
    )
    model = QwenTransformer(args)
    return model, args
