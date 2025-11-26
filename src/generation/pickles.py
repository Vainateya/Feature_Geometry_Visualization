import os
import numpy as np
from src.lsdata.metadata import GPT2LayerMeta
from src.utils import save

from typing import List


def save_layerwise_embeddings(embeddings: np.ndarray, output_dir: str, metadata_template: List[str], skip_norm_capture: bool):
    """
    Saves each layer of tensor as a separate .npy file and corresponding pickled metadata.

    Args:
        embeddings (np.ndarray): Shape (num_layers, n_samples, seq_len, hidden_dim)
        output_dir (str): Save path.
        metadata_template (list[str]): List of labels in order like ["initial", "norm1", ..., "post_mlp"] across all layers
        skip_norm_capture (bool): Whether capture of pre-add norm states were skipped
    """
    os.makedirs(output_dir, exist_ok=True)
    num_layers = embeddings.shape[0]

    if len(metadata_template) != num_layers:
        raise ValueError(f"Expected {num_layers} metadata labels but got {len(metadata_template)}")

    metas = []
    for i in range(num_layers):
        # Save tensor
        npy_path = os.path.join(output_dir, f"{i}.npy")
        np.save(npy_path, embeddings[i])  # (n_samples, seq_len, dim)

        # Generate metadata
        stage = metadata_template[i]
        block_num = -1 if stage == "initial" else (i - 1) // (4 if skip_norm_capture else 6)  # 6 states per block after the initial unless skip_norm_capture, in which case 4

        meta = GPT2LayerMeta(
            id=i,
            block_num=block_num,
            is_embed=(stage == "initial"),
            is_attn=("attn" in stage),
            is_mlp=("mlp" in stage),
            is_norm=("norm" in stage),
            pre_add=("post" not in stage and "initial" not in stage and "norm_f" not in stage),
            vis_color=None,
        )

        # Add meta to metas
        metas.append(meta)

    metas_pkl_path = os.path.join(output_dir, "metadata.pkl")
    save(metas, metas_pkl_path)
