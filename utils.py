def fix_rope(model, model_hf):
    """
    Fix the RoPE parameters for Qwen models in TransformerLens.
    
    This addresses an issue where TransformerLens doesn't correctly set the RoPE parameters
    when loading Qwen models (especially Qwen2.5-Math or Qwen R1 distills), leading to
    incorrect behavior on longer sequences.
    
    Args:
        model: The TransformerLens model
        model_hf: The HuggingFace model
        
    Returns:
        None (modifies the model in-place)
    """
    model.cfg.rotary_base = model_hf.config.rope_theta

    for block in model.blocks:
        attn = block.attn
        sin, cos = attn.calculate_sin_cos_rotary(
            model.cfg.rotary_dim,
            model.cfg.n_ctx,
            base=model.cfg.rotary_base,
            dtype=model.cfg.dtype,
        )
        attn.register_buffer("rotary_sin", sin.to(model.cfg.device))
        attn.register_buffer("rotary_cos", cos.to(model.cfg.device))
        
    return model