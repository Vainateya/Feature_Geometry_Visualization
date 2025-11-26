import torch

def register_gpt2_hooks(model):
    """
    Attaches forward hooks to collect LLaMA layer outputs.

    Args:
        model (torch.nn.Module): The GPT-2 model.

    Returns:
        list[torch.Tensor]: Mutable list populated with outputs
    """
    latents = []
    
    def hook(module, input, output):
        # If output is a tuple, use the first element
        if isinstance(output, tuple):
            output = output[0]
        latents.append(output.detach().cpu())

    for block in model.transformer.h:
        block.attn.register_forward_hook(hook)
        block.ln_1.register_forward_hook(hook)
        block.mlp.register_forward_hook(hook)
        block.ln_2.register_forward_hook(hook)
    
    model.transformer.ln_f.register_forward_hook(hook)
        

    return latents

def register_llama_hooks(model):
    """
    Attaches forward hooks to collect LLaMA layer outputs.

    Args:
        model (torch.nn.Module): The LLaMA model.

    Returns:
        list[torch.Tensor]: Mutable list populated with outputs    
    """
    latents = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        latents.append(output.detach().cpu())

    for block in model.model.layers:
        block.input_layernorm.register_forward_hook(hook)
        block.self_attn.register_forward_hook(hook)
        block.post_attention_layernorm.register_forward_hook(hook)
        block.mlp.register_forward_hook(hook)
    
    model.model.norm.register_forward_hook(hook)

    return latents