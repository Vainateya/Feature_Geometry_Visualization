import os
import shutil
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.utils import get_project_root

HF_MODELS_DIR = os.path.join(get_project_root(), "hf_models")


def load_hf_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads a Hugging Face model and its tokenizer.
    - Uses a local version of the model.
    - Sets the model to evaluation mode with output_hidden_states enabled.

    Args:
        model_name (str): Hugging Face model identifier (currently unused).

    Returns:
        model (nn.Module): The HF model.
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer.
    """
    model_name = model_name.replace("/", "-")
    LOCAL_PATH = os.path.join(HF_MODELS_DIR, f"{model_name}_local")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_PATH, output_hidden_states=True)
    model.eval()
    return model, tokenizer


def save_hf_model(model_name: str, replace_if_exists=False):
    """
    Downloads a Hugging Face model and tokenizer from the Hub and saves them locally 
    under `<project_root>/hf_models/{model_name}_local`.

    Args:
        model_name (str): The original Hugging Face model identifier (e.g., "gpt2" or "meta-llama/Llama-2-7b-hf").
        replace_if_exists (bool): If True, overwrites any existing local copy. Defaults to False.

    Raises:
        Exception: If model or tokenizer saving fails for any reason.

    Returns:
        None
    """
    # Save path — adjust as needed
    safe_name = model_name.replace("/", "-")
    save_path = os.path.join(HF_MODELS_DIR, f"{safe_name}_local")

    already_exists = os.path.exists(save_path)

    if not already_exists or replace_if_exists:
        try:
            # Load from Hugging Face Hub using original model name
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.save_pretrained(save_path)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(save_path)

        except Exception:
            if not already_exists:
                print(
                    f"Error while saving. If it was created, the save directory at {save_path} will be removed since it did not exist before",
                    file=sys.stderr,
                )
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
            else:
                print(
                    f"Error while saving. Save directory at {save_path} will NOT be removed since it existed previously, but check to make sure nothing was corrupted.",
                    file=sys.stderr,
                )

            raise
    else:
        print(f"save_hf_model(): The model {model_name} was not saved as a local copy already exists")
