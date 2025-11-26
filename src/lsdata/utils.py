from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TypeAlias, Union
from tqdm.auto import tqdm

import numpy as np
import os

from src.lsdata.metadata import Metadata, get_attr_list


IndexSelectionType: TypeAlias = Union[
    int,  # Python int
    Sequence[int],  # list/tuple of ints, 1D np.ndarray of ints, for choosing particular indices
    Sequence[bool],  # list/tuple of bools, 1D np.ndarray of bools, for selection via boolean mask
    slice,  # slice object
]


def _load_and_slice(
    file_path: str | os.PathLike,
    sample_selection: IndexSelectionType = slice(None),
    sequence_selection: IndexSelectionType = slice(None),
):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    hidden_state = np.load(file_path, mmap_mode="r")

    # Check hidden state
    if not isinstance(hidden_state, np.ndarray):
        raise ValueError(f"Hidden state from file {file_path} is not a numpy array. Got: {type(hidden_state)}")

    if hidden_state.ndim != 3:
        raise ValueError(
            f"Hidden state from file {file_path} does not have 3 dimensions (representing [n_samples, sequence_length, n_embed]). Got: {hidden_state.ndim}"
        )

    # Slice

    # We copy for several reasons. One is to ensure contiguity of data. The other is that because np.load uses mmap_mode="r", the original load is just a memory map, not an actual load into RAM. The copy() make it so that this method both maps the memory AND loads it in, so that the amount of time this method takes represents the map + load, not just the super fast map.
    hidden_state = hidden_state[sample_selection, sequence_selection].copy()

    if hidden_state.ndim != 3:
        raise ValueError(
            f"Hidden state from file {file_path} did not have 3 dimensions (representing [n_samples, sequence_length, n_embed]) after slicing! Instead it had: {hidden_state.ndim}. This is likely because slicing caused by either sample_selection or sequence_selection resulted in loss of a dimension. Please ensure you are using a supported slicing method!"
        )

    return hidden_state


def load_latent_data(
    dir_path: str | os.PathLike,
    metas: Sequence[Metadata] | None = None,
    latent_ids: Sequence[int] | None = None,
    sample_selection: IndexSelectionType | None = None,
    sequence_selection: IndexSelectionType | None = None,
    verbose: bool = True,
    max_workers: int = None,
):
    """
    Args:
        dir_path (str): Path to a directory which contains in it a file with the name "{latent_id}.pkl" for every latent_id in latent_ids. Each file represents a single hidden state across samples and sequences, and should have shape [n_samples, sequence_length, n_embeds]

        metas (Sequence[Metadata]|None): Sequence of Metadata objects. ids will be extracted from these metas in the order of the sequence, they will then be used to identify the files to be loaded as the latent_ids.
            Either this or latent_ids should be specified, and the other None

        latent_ids (Sequence[int]|None): A sequence of ids specifying which files to load and the contents processed and returned. The files will be returned as an array containing each of the hidden state files stacked along the 0th dimension in the order specified by latent_ids
            Either this or metas should be specified, and the other None

        sample_selection (IndexSelectionType|None): If None, all samples will be used.
            An arugment that will directly be used to slice the desired samples. If int, it will be wrapped in a list first to preserve array rank. Otherwise, it will be passed directly to slice. Please note that 0d arrays (such as 0d numpy arrays) are not supported, as they will not have an extra dimension added before slicing. This will result in an output array with one less dimension than expected.

        sequence_selection (IndexSelectionType|None): Same as sample_selection, except used to select which particular elements of the sequence to slice.

        verbose (bool): Whether to print and use tqdm

        max_workers (int): Max workers for parallelized loading. If None, defaults to os.cpu_count()

    Returns:
        np.ndarray: A 4d array representing the contents of each of the hidden state files stacked along the 0th dimension. If the number of indices specified by sample_selection is n_samples and the number of sequence positions specified by sequence_selection is sequence_length, it will have shape [len(latent_ids) or len(metas), n_samples, sequence_length, n_embed]

    """

    """ Process Args """

    dir_path = Path(dir_path)

    if not (metas is not None) ^ (latent_ids is not None):
        raise ValueError(
            f"Exactly one of metas or latent_ids must be specified. Got metas: {metas} and latent_ids: {latent_ids}"
        )

    if metas is not None:
        latent_ids = get_attr_list(metas, attr_name="id")

    if type(sample_selection) is int:
        sample_selection = [sample_selection]

    if type(sequence_selection) is int:
        sequence_selection = [sequence_selection]

    if sample_selection is None:
        sample_selection = slice(None)

    if sequence_selection is None:
        sequence_selection = slice(None)

    if max_workers is None:
        max_workers = os.cpu_count()

    if len(latent_ids) == 0:
        raise ValueError("There was nothing to load! Length of latent_ids was 0.")

    """ Load files """
    file_paths = [dir_path / f"{latent_id}.npy" for latent_id in latent_ids]

    worker = partial(_load_and_slice, sample_selection=sample_selection, sequence_selection=sequence_selection)

    if verbose:
        print(f"Loading LS Files with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        hidden_states = list(
            tqdm(pool.map(worker, file_paths), total=len(file_paths), desc="Loading LS Files", disable=not verbose)
        )

    if verbose:
        print("Finished loading LS Files")

    return np.stack(hidden_states, axis=0)
