import torch
from tqdm import tqdm
from src.analysis.utils import calc_RMS
from src.generation.hook import register_gpt2_hooks, register_llama_hooks
import numpy as np


def get_hidden_states(model: torch.nn.Module, tokenized_inputs: list[torch.Tensor], model_type: str = "gpt2", skip_norm_capture: bool = False) -> np.ndarray:
    """
    Extracts hidden states from transformer models (e.g., GPT-2, LLaMA) using forward hooks.
    Hooks are registered externally to the model, and are thus non-invasive in nature. 

    Args:
        model: HuggingFace transformer model.
        tokenized_inputs (List[Tensor]): List of (1, seq_len) input tensors.
        model_type (str): Either "gpt2" or "llama".
        skip_norm_capture (bool): Do not capture pre-add norm contributions. Only applicable to pre-norm models

    Returns:
        np.ndarray: Array of shape (num_layers, n_samples, seq_len, hidden_dim).

        Note here that num_layers “num_layers” here refers to the full unfolded latent sequence (not the transformer block count)
        since each block contributes multiple latent states (pre-attn norm, pre-mlp norm, attn, mlp, post-add attn, post-add mlp )
    """
    if model_type.lower() == "gpt2":
        latents = register_gpt2_hooks(model)
    elif "llama" in model_type.lower():
        latents = register_llama_hooks(model)
    else:
        raise ValueError(f"Unsupported model type '{model_type}' for latent extraction.")

    all_latents = []

    with torch.no_grad():
        for input_ids in tqdm(tokenized_inputs, desc="Extracting Hidden States"):
            latents.clear()
            outputs = model(input_ids)
            initial_embedding = outputs.hidden_states[0].detach().cpu()

            # Build sequence: [initial, norm1, attn, post_attn, norm2, mlp, post_mlp] * n
            full_sequence = [initial_embedding]

            # Determine number of transformer blocks dynamically based on model architecture
            if hasattr(model, "transformer"):  # e.g., GPT2
                num_blocks = len(model.transformer.h)
            elif hasattr(model, "model"):  # e.g., LLaMA
                num_blocks = len(model.model.layers)
            else:
                raise ValueError("Cannot determine number of transformer blocks for this model.")

            assert len(latents) == ((4 * num_blocks) + 1), f"Expected {4 * num_blocks + 1} hooks, got {len(latents)}"

            for i in range(num_blocks):
                norm1 = latents[4 * i + 0]
                attn = latents[4 * i + 1]
                norm2 = latents[4 * i + 2]
                mlp = latents[4 * i + 3]

                prev = full_sequence[-1]

                post_attn = prev + attn
                post_mlp = post_attn + mlp

                if not skip_norm_capture:
                    full_sequence.extend([norm1, attn, post_attn, norm2, mlp, post_mlp])
                else:
                    full_sequence.extend([attn, post_attn, mlp, post_mlp])


            final_norm = latents[-1]
            full_sequence.append(final_norm)

            # This should be the POST-ADD hidden state in the second-to-last transformer block
            output_second_last_LS = outputs.hidden_states[-2].detach().cpu()
            # SHOULD be the same as output_second_last_LS above if we extraced and added things correctly
            extracted_second_last_LS = full_sequence[-8 if not skip_norm_capture else -6]
            if not torch.allclose(extracted_second_last_LS, output_second_last_LS, equal_nan=True):
                raise ValueError(
                    f"The latent state tensor of the second-to-last transformer block generated via hooks was not close to the actual latent state directly obtain from model output! The RMSE between the tensors was: {calc_RMS(extracted_second_last_LS - output_second_last_LS)}"
                )

            stacked = torch.stack(full_sequence, dim=0)
            all_latents.append(stacked)

    latents.clear()

    final_tensor = torch.cat(all_latents, dim=1)
    return final_tensor.numpy()
