import os
import argparse
import warnings
import torch
import numpy as np
from src.generation.loader import load_hf_model, save_hf_model
from src.generation.data import load_dataset_for_gen, load_single_tokens, tokenize, load_repeated_tokens
from src.generation.pickles import save_layerwise_embeddings
from src.generation.extract import get_hidden_states
from src.generation.utils import select_best_device
from src.generation.permute import get_permutation
from src.utils import get_project_root


def run_pipeline(
    model_name: str,
    mode: str,
    num_inputs: int | None,
    output_dir: str,
    dataset_name: str | None = None,
    sequence_length: int | None = None,
    skip_norm_capture: bool = False,
    **kwargs,
):
    """
    Runs the latent extraction pipeline: model + tokenizer loading, data loading,
    hidden state extraction, and layerwise saving.

    Args:
        model_name (str): Model name or path to local HF model.
        mode (str): One of "text", "singular", or "repeat"
        num_inputs (int): Number of inputs (samples) to process.
        output_dir (str): Directory to save extracted layerwise embeddings.
        dataset_name (str | None): Name of dataset to use. Only used for "text" mode
        sequence_length (int | None): Length of each input sequence. If None uses maximum supported by model. Not applicable for "singular" mode
        skip_norm_capture (bool): Do not capture pre-add norm contributions. Only applicable to pre-norm models
    """

    if mode != "text" and dataset_name is not None:
        raise ValueError(f"Specifying a dataset name is not allowed for mode {mode}")

    save_hf_model(model_name)
    device = select_best_device(mode="m")
    print(f"Using device: {device}")

    model, tokenizer = load_hf_model(model_name)
    model.to(device)
    tokenizer.padding_side = "right"

    permute_strategy = kwargs.get("permute", "identity").lower()
    if permute_strategy != "identity":
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            blocks = model.transformer.h
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            blocks = model.model.layers
        else:
            warnings.warn("Permutation requested but model layers not found.")
            blocks = None

        if blocks is not None:
            perm = get_permutation(permute_strategy, len(blocks))
            original = [blocks[i] for i in perm]
            for i, b in enumerate(original):
                blocks[i] = b
            print(f"Applied permutation: {perm}")

    max_context = getattr(model.config, "n_positions", model.config.max_position_embeddings)

    if mode == "singular":
        sequence_length = 1
    elif sequence_length is None:
        sequence_length = max_context
    elif sequence_length > max_context:
        warnings.warn(
            f"Sequence length {sequence_length} was greater than maximum supported by {model_name} ({max_context}). Using max supported length."
        )
        sequence_length = max_context

    num_blocks = getattr(model.config, "n_layer", model.config.num_hidden_layers)
    if not skip_norm_capture:
        metadata_template = (
            ["initial"] + ["norm_attn", "attn", "post_attn", "norm_mlp", "mlp", "post_mlp"] * num_blocks + ["norm_f"]
        )
    else:
        metadata_template = (
            ["initial"] + ["attn", "post_attn", "mlp", "post_mlp"] * num_blocks + ["norm_f"]
        )

    if mode == "text":
        inputs = load_dataset_for_gen(dataset=dataset_name, num_samples=num_inputs)
        tokenized = tokenize(dataset=inputs, tokenizer=tokenizer, device=device, max_length=sequence_length, add_special_tokens=True)
    elif mode == "repeat":
        repeat_token_id = kwargs.get("repeat_token_id", None)
        vocab_size = len(tokenizer)
        if not (0 <= repeat_token_id < vocab_size):
            raise ValueError(f"--repeat_token_id {repeat_token_id} is out of range [0, {vocab_size - 1}]")
        if sequence_length is None or sequence_length < 1:
            raise ValueError("mode='repeat' requires a positive --sequence_length.")
        if repeat_token_id is None:
            raise ValueError("mode='repeat' requires --repeat_token_id <int>.")
        if num_inputs is None:
            raise ValueError("mode='repeat' requires --num_inputs to specify batch size.")
        tokenized = load_repeated_tokens(
            tokenizer=tokenizer,
            device=device,
            token_id=repeat_token_id,
            seq_len=sequence_length,
            batch_size=num_inputs,
        )
    elif mode == "singular":
        if num_inputs is None:
            vocab_size = len(tokenizer.get_vocab())
            print(f"No num_inputs specified. Using full vocabulary size: {vocab_size}")
        else:
            vocab_size = num_inputs
        tokenized = load_single_tokens(tokenizer=tokenizer, device=device, vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Loaded {len(tokenized)} inputs.")

    # Capture token IDs and string versions
    token_id_array = torch.cat(tokenized, dim=0).cpu().numpy()
    token_str_array = np.array([tokenizer.convert_ids_to_tokens(seq[0].cpu().tolist()) for seq in tokenized])

    expected_batch = len(tokenized)
    expected_seq = tokenized[0].shape[1] if expected_batch > 0 else sequence_length

    assert token_id_array.shape == (expected_batch, expected_seq), (
        f"Expected token_id_array shape {(expected_batch, expected_seq)}, but got {token_id_array.shape}"
    )
    assert token_str_array.shape == (expected_batch, expected_seq), (
        f"Expected token_str_array shape {(expected_batch, expected_seq)}, but got {token_str_array.shape}"
    )

    latent_tensor = get_hidden_states(model, tokenized, model_type=model_name, skip_norm_capture=skip_norm_capture)
    print(f"Latent tensor shape: {latent_tensor.shape}")

    save_layerwise_embeddings(latent_tensor, output_dir, metadata_template, skip_norm_capture=skip_norm_capture)
    np.save(os.path.join(output_dir, "token_ids.npy"), token_id_array)
    np.save(os.path.join(output_dir, "token_strings.npy"), token_str_array)


def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)


def parse_args():
    """Parse command‑line arguments for the latent‑extraction pipeline."""

    parser = argparse.ArgumentParser(description="Run the latent‑extraction pipeline over a set of input arguments.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        help='Either "text" or "singular" or "repeat" (repeat requires --repeat_token_id)',
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help='Only valid for "text" mode. Choose from "pg19" or "tinystories"',
    )
    parser.add_argument(
        "--num_inputs",
        type=none_or_int,
        default=64,
        help="Number of samples to use. Use 'None' to select all (only valid for 'singular' mode).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save extracted latents. If omitted, a directory is auto‑constructed.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=None,
        help="Length of input sequences. None will use sequence length corresponding to model max context length (for applicable modes)",
    )
    parser.add_argument(
        "--permute",
        type=str,
        default="identity",
        help="Permutation strategy for transformer blocks: 'identity', 'random', 'reverse', 'swap:i,j', 'custom:...'",
    )
    parser.add_argument(
        "--repeat_token_id",
        type=int,
        default=None,
        help="Required when --mode=repeat. The token id to repeat to fill each sequence.",
    )
    parser.add_argument(
        "--skip_norm_capture",
        type=bool,
        default=False,
        help="Do not capture pre-add norm contributions (only applies to pre-norm models).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Derive an output directory if the user didn't supply one explicitly.
    repeat_suffix = (
        f"-token{args.repeat_token_id}" if args.mode == "repeat" and args.repeat_token_id is not None else ""
    )
    output_dir = args.output_dir or os.path.join(
        get_project_root(),
        "processed_data",
        f"{args.model_name.replace('/', '-')}_latents-{args.mode}{repeat_suffix}-{args.dataset + '-' if args.dataset else ''}{args.num_inputs if args.num_inputs is not None else 'full'}_samples-{args.sequence_length}_sequence_length-{args.permute}",
    )

    run_pipeline(
        model_name=args.model_name,
        mode=args.mode,
        dataset_name=args.dataset,
        num_inputs=args.num_inputs,
        output_dir=output_dir,
        sequence_length=args.sequence_length,
        permute=args.permute,
        repeat_token_id=args.repeat_token_id,
        skip_norm_capture=args.skip_norm_capture,
    )


if __name__ == "__main__":
    main()
