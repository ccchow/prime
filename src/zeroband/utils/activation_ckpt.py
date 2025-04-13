import torch.nn as Module
from torch.nn import ModuleList # Add import for ModuleList

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from zeroband.utils.logger import get_logger


def apply_ac_ckpt(model: Module, num: int):
    """Apply activation checkpointing to the model.
    Supports both zeroband internal models and Hugging Face models.
    Apply to layers multiple of `num`.

    Example if `num=2` only half of the layers are checkpointed.
    """
    logger = get_logger()
    layer_container = None

    # Try common attributes for layers in HF models and zeroband models
    if hasattr(model, 'layers'): # zeroband Llama, some HF models
        layer_container = model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): # GPT-2 style HF models
        layer_container = model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'): # Other HF model structures
        layer_container = model.model.layers
    else:
        logger.warning("Could not find standard layer container ('layers', 'transformer.h', 'model.layers'). Activation checkpointing not applied.")
        return model # Return original model if layers not found

    if not hasattr(layer_container, 'named_children') and not isinstance(layer_container, ModuleList):
        logger.warning(f"Layer container type {type(layer_container)} not directly supported for named iteration. Activation checkpointing may not be applied correctly.")
        # Attempt iteration anyway if possible, otherwise return
        try:
            # Try iterating with enumerate if it's a list/sequence
            iterable_layers = list(enumerate(layer_container))
        except TypeError:
            logger.error("Cannot iterate over the found layer container.")
            return model
    else:
        # Prefer named_children if available, otherwise assume ModuleList index access
        iterable_layers = layer_container.named_children() if hasattr(layer_container, 'named_children') else enumerate(layer_container)


    applied_count = 0
    # Use index for modification if it's a ModuleList or list
    is_list_like = isinstance(layer_container, (ModuleList, list))

    for idx, layer in iterable_layers:
        # Use idx directly if it's from enumerate, or convert layer_id if from named_children
        current_index = idx if isinstance(idx, int) else int(idx) # Assuming layer_id from named_children is string convertible to int

        if current_index % num == 0:
            wrapped_layer = checkpoint_wrapper(layer, preserve_rng_state=False)
            if is_list_like:
                layer_container[current_index] = wrapped_layer
            elif hasattr(layer_container, 'register_module'): # Handles ModuleDict or custom containers with register_module
                layer_container.register_module(str(idx), wrapped_layer) # Use original idx (could be string)
            else:
                # Fallback: Try setting attribute directly if possible (less common)
                try:
                    setattr(layer_container, str(idx), wrapped_layer)
                except AttributeError:
                    logger.error(f"Could not apply checkpoint wrapper for layer {idx}. Container type {type(layer_container)} does not support item assignment or register_module.")
                    continue # Skip this layer if cannot modify
            applied_count += 1

    if applied_count > 0:
        logger.debug(f"Applied activation checkpointing to {applied_count} layers")
    else:
        logger.warning("Activation checkpointing was enabled, but no layers were wrapped. Check model structure and layer container.")

    # No explicit return needed as model is modified in-place, but returning it is fine
    return model
