from dataclasses import dataclass
from collections.abc import Callable, Sequence
from typing import Any


@dataclass
class Metadata:
    """Generic metadata class"""

    id: int  # Unique identifying ID


@dataclass
class StandardTransformerMeta(Metadata):
    """Class for keeping track of attributes from a particular latent state of a 'standard' transformer model which help to identify the latent state and any other useful properties

    Works for both pre-norm and post-norm models, although the meanings will be a little different between.
    """

    block_num: int
    is_attn: bool
    is_mlp: bool
    is_embed: bool
    is_norm: bool
    pre_add: bool
    vis_color: Any = None


@dataclass
class GPT2LayerMeta(StandardTransformerMeta):
    """Same as StandardTransformerMeta. Exists for legacy reasons"""

    pass


def get_attr_list(metas: Sequence[Metadata], attr_name: str) -> list:
    """
    Args:
        metas (Sequence[Metadata]): A sequence containing Metadata objects
        attr_name (str): The name of the desired attribute

    Returns:
        list: A of the specified attribute from each meta object in metas, in the order of their respective objects from metas
    """

    """ Validating Args """
    for meta in metas:
        if not isinstance(meta, Metadata):
            raise TypeError(f"Expected Metadata object, got {type(meta)}")
        if not hasattr(meta, attr_name):
            raise AttributeError(f"Meta object {meta} does not have attribute '{attr_name}'")

    """ Extract Attribute """

    attr_list = [getattr(meta, attr_name) for meta in metas]

    return attr_list


def filter_metas(metas: Sequence[Metadata], layer_filter: Callable[[Metadata], bool]) -> list[Metadata]:
    """
    Args:
        metas (Sequence[Metadata]): A sequence of Metadata objects

        layer_filter (Callable[[Metadata], bool]): A function that can take the metadata objects and return whether or not it should be kept

    Returns:
        list[Metadata]: A list of the Metadata objects that were kept by layer_filter
    """

    for meta in metas:
        if not isinstance(meta, Metadata):
            raise TypeError(f"Expected Metadata object, got {type(meta)}")

    filtered_metas = [meta for meta in metas if layer_filter(meta)]

    return filtered_metas


def sort_metas_by_id(metas: Sequence[Metadata]) -> list[Metadata]:
    """
    Args:
        metas (Sequence[Metadata]): A sequence of Metadata objects

    Returns:
        list[Metadata]: A new list of the Metadata objects sorted by id (increasing)
    """

    return sorted(metas, key=lambda meta: meta.id)
