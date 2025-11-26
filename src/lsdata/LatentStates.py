from __future__ import annotations

from collections.abc import Callable, Sequence
import copy
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT

from src.analysis.utils import convert_unit_vectors
from src.lsdata.metadata import Metadata, filter_metas, get_attr_list
from src.lsdata.utils import IndexSelectionType, load_latent_data
from src.utils import load, save


class LSData:
    """
    A class for storing and manipulating latent state data from transformers.

    LSData encapsulates:
    - Transformer latent states (data)
    - Metadata describing each layer (metas)
    - Token IDs and strings for the input sequences

    The class supports loading data from disk or direct instantiation with numpy arrays.
    It provides methods for data manipulation like unit vector conversion, sorting,
    and filtering operations.

    Attributes:
        dir_path (Path | None): Path to directory containing saved LSData files. Specify either this or data, metas, token_ids, and token_strings
        data (NDArray): Transformer activations with shape (n_layers, n_samples, seq_len, n_features)
        metas (Sequence[Metadata]): Metadata objects describing each layer
        token_ids (NDArray): Token IDs with shape (n_samples, seq_len)
        token_strings (NDArray): Token strings with shape (n_samples, seq_len)

    Args:
        dir_path (str | os.PathLike | None): Path to directory containing saved LSData files. Mutually exclusive with (data, metas, token_ids, token_strings)
        layer_filter (Callable[[Metadata], bool] | None): Optional function to filter layers based on metadata
        sample_selection (IndexSelectionType | None): Optional indexing to select specific samples
        sequence_selection (IndexSelectionType | None): Optional indexing to select specific sequence positions
        data (NDArray | None): Transformer activations with shape (n_layers, n_samples, seq_len, n_features). Must be provided with metas, token_ids, and token_strings
        metas (Sequence[Metadata] | None): Metadata objects describing each layer. Must be provided with data, token_ids, and token_strings
        token_ids (NDArray | None): Token IDs with shape (n_samples, seq_len). Must be provided with data, metas, and token_strings
        token_strings (NDArray | None): Token strings with shape (n_samples, seq_len). Must be provided with data, metas, and token_ids
        verbose (bool): Whether to print progress information
        max_workers (int | None): Maximum number of worker threads for parallel loading
    """

    def __init__(
        self,
        dir_path: str | os.PathLike | None = None,
        layer_filter: Callable[[Metadata], bool] | None = None,
        sample_selection: IndexSelectionType | None = None,
        sequence_selection: IndexSelectionType | None = None,
        data: NDArray | None = None,
        metas: Sequence[Metadata] | None = None,
        token_ids: NDArray | None = None,
        token_strings: NDArray | None = None,
        verbose: bool = True,
        max_workers: int | None = None,
    ):
        """Arg Checking"""

        if not (
            (dir_path is not None)
            ^ (data is not None and metas is not None and token_ids is not None and token_strings is not None)
        ):
            raise ValueError(
                f"{self.__class__.__name__}: Either dir_path must be provided, or (data, metas, token_ids, and token_strings) must be provided. Not both nor neither."
            )

        if (data is not None and metas is not None and token_ids is not None and token_strings is not None) and (
            layer_filter is not None or sample_selection is not None or sequence_selection is not None
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__}: Filtering, sample selection, and sequence selection are not yet implemented for directly instantiation from data, metas, token_ids, and token_strings. Please provide dir_path instead."
            )

        """ Initializing """

        self.dir_path = dir_path

        if self.dir_path is not None:
            self.dir_path = Path(self.dir_path)
            self._init_from_path(
                layer_filter=layer_filter,
                sample_selection=sample_selection,
                sequence_selection=sequence_selection,
                verbose=verbose,
                max_workers=max_workers,
            )
        else:
            self.metas = metas
            self.data = data
            self.token_ids = token_ids
            self.token_strings = token_strings

        self._validate()

    def _validate(self):
        if self.data.ndim != 4:
            raise ValueError(
                f"data did not have 4 dimensions! Instead it had {self.data.ndim} dimensions, and its shape was: {self.data.shape}"
            )
        if self.token_ids.ndim != 2:
            raise ValueError(
                f"token_ids did not have 2 dims (n_samples, sequence_len). Instead it had shape: {self.token_ids.shape}"
            )
        if self.token_strings.ndim != 2:
            raise ValueError(
                f"token_strings did not have 2 dims (n_samples, sequence_len). Instead it had shape: {self.token_ids.shape}"
            )

        n_layers, n_samples, sequence_length, hidden_size = self.data.shape

        if len(self.metas) != n_layers:
            raise ValueError(
                f"metas must have the same length as dim 0 of data! len(metas): {len(self.metas)}, data.shape: {self.data.shape}"
            )

        if self.token_ids.shape != (n_samples, sequence_length):
            raise ValueError(
                f"token_ids did not have the expected shape (n_samples, sequence_len). From the data, this shape should be {(n_samples, sequence_length)}. But instead it was: {self.token_ids.shape}"
            )
        if self.token_strings.shape != (n_samples, sequence_length):
            raise ValueError(
                f"token_strings did not have the expected shape (n_samples, sequence_len). From the data, this shape should be {(n_samples, sequence_length)}. But instead it was: {self.token_strings.shape}"
            )

    def _init_from_path(
        self,
        layer_filter: Callable[[Metadata], bool] = None,
        sample_selection: IndexSelectionType | None = None,
        sequence_selection: IndexSelectionType | None = None,
        verbose: bool = True,
        max_workers: int = None,
    ):
        if self.dir_path is None:
            raise ValueError(f"{self.__class__.__name__}: dir_path must be provided to initialize from path.")

        """ Obtain metas """
        # Unpickle
        metas = load(self.dir_path / "metadata.pkl")

        # Filter
        if layer_filter is not None:
            metas = filter_metas(metas=metas, layer_filter=layer_filter)
        self.metas = metas

        """ Get data """
        self.data = load_latent_data(
            dir_path=self.dir_path,
            metas=self.metas,
            sample_selection=sample_selection,
            sequence_selection=sequence_selection,
            verbose=verbose,
            max_workers=max_workers,
        )

        """ Get token data """

        if type(sample_selection) is int:
            sample_selection = [sample_selection]

        if type(sequence_selection) is int:
            sequence_selection = [sequence_selection]

        if sample_selection is None:
            sample_selection = slice(None)

        if sequence_selection is None:
            sequence_selection = slice(None)

        token_ids = np.load(self.dir_path / "token_ids.npy", mmap_mode="r")
        token_strings = np.load(self.dir_path / "token_strings.npy", mmap_mode="r")

        self.token_ids = token_ids[sample_selection, sequence_selection].copy()
        self.token_strings = token_strings[sample_selection, sequence_selection].copy()

    def convert_to_unit_vector(self):
        """
        Returns:
            LSData: A new object with the data converted to unit vectors. The metas, token_ids, and token_strings are deepcopied so the references are not shared.
        """
        return LSData(
            data=convert_unit_vectors(self.data),
            metas=copy.deepcopy(self.metas),
            token_ids=copy.deepcopy(self.token_ids),
            token_strings=copy.deepcopy(self.token_strings),
        )

    def copy_sort(self, key: Callable[[Metadata], SupportsRichComparisonT]) -> "LSData":
        """
        Returns a copy of self with both metas and data layers sorted according to key.

        Args:
            key (Callable[[Metadata], SupportsRichComparisonT]): A function that takes a Metadata object and returns a value that can be used for sorting comparison.

        Returns:
            LSData: A new LSData object with metas and corresponding data layers sorted according to the key function.
        """
        new_lsdata = copy.deepcopy(self)

        # Save original idx to each meta
        for i, meta in enumerate(new_lsdata.metas):
            meta._copy_sort_idx = i

        # Sort metas
        new_lsdata.metas = sorted(new_lsdata.metas, key=key)

        # Get now permuted list of original idx
        permuted_idxs = get_attr_list(new_lsdata.metas, "_copy_sort_idx")

        # Remove the temp attribute from each metas
        for meta in new_lsdata.metas:
            del meta._copy_sort_idx

        # Permute the layers of data in the same way the metas were permutted
        new_lsdata.data = new_lsdata.data[permuted_idxs]

        return new_lsdata


def save_lsdata(lsdata: LSData, save_dir_path: str | os.PathLike, exists_ok: bool = True, verbose: bool = False):
    """
    Given an LSData object, save it to the specified directory such that it can be loaded again later.

    Args:
        lsdata (LSData): The LSData object to save

        save_dir_path (str | os.PathLike): The directory to save the data in

        exists_ok (bool): Whether or not the directory in which to save the data can already exist. If True, saving may overwrite existing files with the same names.
    """

    save_dir_path = Path(save_dir_path)

    if save_dir_path.exists():
        if not exists_ok:
            raise ValueError(f"exists_ok was False but the path '{save_dir_path}' already exists!")
        if not save_dir_path.is_dir():
            raise ValueError(
                f"The path {save_dir_path} already exists but it is not a directory! The path must be to a directory where the files will be saved"
            )
    else:
        save_dir_path.mkdir()

    # Validate the LSData instance
    lsdata._validate()

    # Get everything to save
    metas = lsdata.metas
    data = lsdata.data
    token_ids = lsdata.token_ids
    token_strings = lsdata.token_strings

    # Save the metas
    metas_path = save_dir_path / "metadata.pkl"
    save(metas, metas_path)

    # Save the data
    for meta, layer_data in tqdm(zip(metas, data), desc="Saving Layers", disable=not verbose):
        layer_data_path = save_dir_path / f"{meta.id}.npy"
        np.save(layer_data_path, layer_data)  # (n_samples, seq_len, dim)

    # Save token_ids and token_strings
    np.save(save_dir_path / "token_ids.npy", token_ids)
    np.save(save_dir_path / "token_strings.npy", token_strings)
