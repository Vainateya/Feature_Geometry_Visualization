import os
import numpy as np
import torch
from dataclasses import dataclass
from typing import Callable, Sequence
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor

from src.lsdata.metadata import Metadata, StandardTransformerMeta
from src.lsdata.utils import IndexSelectionType


def convert_unit_vectors(x: np.ndarray | torch.Tensor, in_place: bool = False) -> np.ndarray | torch.Tensor:
    """
    Given either a numpy array or torch tensor, converts to unit vectors by treating the last dimension as the vector dimension

    Args:
        x (np.ndarray|torch.Tensor): Input array or tensor
        in_place (bool): If True, performs the modification in-place. Default False

    Returns:
        np.ndarray|torch.Tensor: Unit vectors of the input. Array or tensor has same shape as input
    """

    if isinstance(x, np.ndarray):
        magnitudes = np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True))
    elif isinstance(x, torch.Tensor):
        magnitudes = torch.sqrt(torch.sum(torch.square(x), dim=-1, keepdim=True))
    else:
        raise ValueError(f"x was not a numpy array nor torch tensor! Instead it was type: {type(x)}")

    if not in_place:
        return x / magnitudes
    else:
        x /= magnitudes
        return x


def apply_layer_norm(x: np.ndarray | torch.Tensor, eps: float = 1e-5) -> np.ndarray | torch.Tensor:
    """
    Given a numpy array or torch tensor, applies layernorm (without any learned gamma/beta) across the last dim.

    Args:
        x (np.ndarray | torch.Tensor): Input array or tensor
        eps (float): Small denominator term for numerical stability

    Returns:
        np.ndarray | torch.Tensor: array or tensor with same shape as input
    """

    if isinstance(x, np.ndarray):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        sqrt = np.sqrt
    elif isinstance(x, torch.Tensor):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, correction=0, keepdim=True)
        sqrt = torch.sqrt
    else:
        raise ValueError(f"x was not a numpy array nor torch tensor! Instead it was type: {type(x)}")

    return (x - mean) / sqrt(var + eps)


def calc_RMS(
    x: np.ndarray | torch.Tensor,
    axis: None | int | tuple[int] = None,
    dim: None | int | tuple[int] = None,
    keepdims: bool | None = None,
    keepdim: bool | None = None,
) -> np.ndarray | torch.Tensor:
    """
    Given either a numpy array or torch tensor, calculates the root mean square.

    dim or axis are interchangable. If either one is specified, it will be provided to a numpy as the 'axis' argument or to torch as a the 'dim' arugment. They will be used for the mean operation, since that is a reducing operation. keepdims (numpy) and keepdim (torch) are similarly interchangeable.

    Args:
        x (np.ndarray | torch.Tensor): Numpy array or torch tensor
        axis (None | int | tuple[int] ): 'axis' argument for numpy or 'dim' argument for torch
        dim (None | int | tuple[int] ): 'axis' argument for numpy or 'dim' argument for torch
        keepdims (None | bool): 'keepdims' argument for numpy or 'keepdim' argument for torch. If None will internally default to False
        keepdim (None | bool): 'keepdims' argument for numpy or 'keepdim' argument for torch. If None will internally default to False


    Returns:
        np.ndarray | torch.Tensor: Root mean square result
    """

    if axis is not None and dim is not None:
        raise ValueError(f"Only one of axis or dim can be specified! Not both. Got axis = {axis}, dim = {dim}")

    if keepdims is not None and keepdim is not None:
        raise ValueError(
            f"Only one of keepdims or keepdim can be specified! Not both. Got keepdims = {keepdims}, keepdim = {keepdim}"
        )

    if dim is not None:
        axis = dim

    if keepdim is not None:
        keepdims = keepdim

    if keepdims is None:
        keepdims = False

    if isinstance(x, np.ndarray):
        return np.sqrt(np.mean(np.square(x), axis=axis, keepdims=keepdims))
    elif isinstance(x, torch.Tensor):
        return x.square().mean(dim=axis, keepdim=keepdims).sqrt()
    else:
        raise ValueError(f"x was not a numpy array nor torch tensor! Instead it was type: {type(x)}")


@dataclass
class Vis2DDimReductConfig:
    """
    Configuration for src.analysis.visualize.vis_2d_dim_reduct.

    Pass an instance of this config directly to vis_2d_dim_reduct.

    Args:
        # Data processing options
        convert_unit_vector (bool, optional): If True, normalize each data point to unit length. Defaults to False.
        convert_layer_norm (bool, optional): If True, apply layer normalization to the data. Defaults to False.

        # Averaging options
        mean_colors (bool, optional): If True, average data points that share the same vis_color. Defaults to False.
        mean_layers (bool, optional): If True, average across the layer dimension. All layers will be given the same vis color (black). Defaults to False.
        mean_samples (bool, optional): If True, average across the sample dimension. Defaults to False.
        mean_seq (bool, optional): If True, average across the sequence dimension. Defaults to False.

        # Dimensionality reduction options
        dim_reduct_mode (str | None, optional): The dimensionality reduction method to use.
            Must be one of "pca" or "umap". If None, prev_dim_reduct must be provided. Defaults to "pca".
        prev_dim_reduct (PCA | UMAP | None, optional): A pre-fitted dimensionality reduction model to use.
            If provided, no new model will be fitted. Defaults to None.
        n_components (int | None, optional): Number of components for dimensionality reduction.
            If None, uses default for the chosen method. Defaults to None.
        dim_reduct_kwargs (dict | None):
            Kwargs for the dim reduct object
        n_fit (float | int | None):
            Settings to fit dim_reduct to only a random subset of the data. If it is a float (in the interval (0, 1) ), the fit will be to that proportion of the data. If an int, it will be to that number of datapoints.
        n_transform (float | int | None):
            Settings to transform and visualize only a random subset of data. If it is a float (in the interval (0, 1) ), the fit will be to that proportion of the data. If an int, it will be to that number of datapoints. The number of datapoints transformed/visualized may not be exactly equal to n_transform. Instead, it will be the closest value which allows an equal number of data points per layer.
        disable_pca_transform_centering (bool, optional): If True and using PCA, disables centering during transform.
            Only affects PCA, ignored for UMAP. Defaults to True.

        # Visualization options
        vis_dims (tuple[int, int] | Sequence[tuple[int, int]] | None, optional): Which dimensions to visualize.
            Can be a single tuple (x_dim, y_dim) or sequence of tuples for multiple plots. Defaults to None.
        vis_lims (dict | None, optional): Dictionary mapping dimension tuples to ((x_min, x_max), (y_min, y_max))
            for setting axis limits. Use None for auto-scaling. Defaults to None.

        # Figure options
        figsize (tuple[int, int] | None, optional): Figure size as (width, height) in inches. Defaults to None.
        dpi (int | None, optional): Figure resolution in dots per inch. Defaults to None.
        title (str | None, optional): Plot title. The dimensionality reduction method will be appended. Defaults to "".
        s (float | ArrayLike, optional): Marker size for scatter plot points. Defaults to 0.01.
        fontsize (float | int | str, optional): Font size for axis labels and title. Defaults to 10.
        labelsize (float | int | str, optional): Font size for tick marks. Defaults to 10.
        subplots_adjust_kwargs (dict[str, float | None] | None): kwargs for fig.subplots_adjust(...) in vis_scatter(...) function. If None, then fig.subplots_adjust will not be called
        disable_labels (bool): Disables x and y axis labels
        disable_ticks (bool): Disables tick marks and tick labels on both axes
        yaxis_formatter (plt.FuncFormatter | str | Callable | None): Function to format y-axis labels. If None, then y-axis labels will not be formatted.
        sequence_cmap (str | mcolor.Colormap | None): Colormap to use for the sequence positions only in the case that mean_layers is True. Otherwise colors correspond to layers, not sequence position. If None then all points will be black. At the moment this is also incompatible with usage of n_transform, as usage of n_transform does not guarentee that there are the same number of samples per sequence position, making the sequence cmap process, which relies on giving all samples at a sequence position the same color, undefined. If n_transform is not None then all points will be black.
        reverse_sequence_cmap (bool): Reverses the order of sequence_cmap in the event it is used

        # Control options
        verbose (bool, optional): If True, print warnings and progress information. Defaults to True.
        show_image (bool): Whether to show the image(s)
        return_images (bool, optional): If True, return the generated plot images. Defaults to False.
        gpu_accelerate (bool): If True, will attempt to use GPU accelerated versions of some more intensive dim reduct methods such as UMAP. If cuda is not available or the proper packages are not installed, will throw a warning and default to False (non-accelerated). Not all dim reduct methods have a GPU accelerated version implemented because some, like PCA, are very fast already.
        seed (int | None): Seed for rng
    """

    # Data processing options
    convert_unit_vector: bool = False
    convert_layer_norm: bool = False

    # Averaging options
    mean_colors: bool = False
    mean_layers: bool = False
    mean_samples: bool = False
    mean_seq: bool = False

    # Dimensionality reduction options
    dim_reduct_mode: str | None = "pca"
    n_components: int | None = None
    dim_reduct_kwargs: dict | None = None
    n_fit: float | int | None = None
    n_transform: float | int | None = None
    disable_pca_transform_centering: bool = True

    # Visualization options
    vis_dims: tuple[int, int] | Sequence[tuple[int, int]] | None = None
    vis_lims: dict[tuple[int, int], tuple[tuple[float, float] | None, tuple[float, float] | None]] | None = None

    # Figure options
    figsize: tuple[int, int] | None = None
    dpi: int | None = None
    title: str | None = ""
    s: float | ArrayLike = 0.01
    fontsize: float | int | str = 10
    labelsize: float | int | str = 10
    subplots_adjust_kwargs: dict[str, float | None] | None = None
    disable_labels: bool = False
    disable_ticks: bool = False
    yaxis_formatter: plt.FuncFormatter | str | Callable | None = None
    sequence_cmap: str | mcolor.Colormap | None = "winter"
    reverse_sequence_cmap: bool = False

    # Control options
    verbose: bool = True
    show_image: bool = False
    gpu_accelerate: bool = True
    seed: int | None = None


def default_colorizer(meta: StandardTransformerMeta):
    if meta.is_mlp and not meta.is_norm:
        meta.vis_color = "A"
    elif meta.is_attn and not meta.is_norm:
        meta.vis_color = "B"
    elif meta.is_embed:
        meta.vis_color = "C"
    elif meta.is_norm and meta.is_mlp:
        meta.vis_color = "D"
    elif meta.is_norm and meta.is_attn:
        meta.vis_color = "E"
    elif meta.is_norm and not meta.pre_add:
        meta.vis_color = "F"
    else:
        raise ValueError("Meta colorization was not defined under current rules! Meta:\n", meta)


@dataclass
class VisualizeConfig:
    ## Data
    data_name: str = None
    layer_filter: Callable[[Metadata], bool] | None = None
    sample_selection: IndexSelectionType | None = None
    sequence_selection: IndexSelectionType | None = None

    ## Visualization

    colorizer: Callable[[Metadata], None] | None = default_colorizer

    prev_vis_mode: int = 1

    do_many_vis: bool = False
    composite_vis_grid_shape: tuple[int, int] = (3, 5)
    composite_img_padding: int = 0

    ## Other
    # for data loading
    img_save_dir: os.PathLike | None = None
    max_workers: int | None = None
    font_family: str | None = "Nimbus Roman"
