from dataclasses import replace
import io
from pathlib import Path
from typing import Any, Callable, Sequence
import warnings
from IPython.display import display
from matplotlib.typing import ColorType
import torch
import numpy as np
from numpy.typing import ArrayLike
from sklearn.decomposition import PCA
from umap import UMAP

try:
    from cuml import UMAP as cuml_UMAP

    HAS_CUML = True
except ImportError:
    cuml_UMAP = None
    HAS_CUML = False

import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
from PIL import Image, ImageOps


from src.lsdata.LatentStates import LSData
from src.lsdata.metadata import Metadata, filter_metas, sort_metas_by_id
from src.lsdata.utils import get_attr_list
from src.analysis.utils import Vis2DDimReductConfig, VisualizeConfig, apply_layer_norm, convert_unit_vectors
from src.lsdata.metadata import StandardTransformerMeta
from src.utils import get_project_root, timer

DARK_SHADE_FACTOR = 0.5
LIGHT_SHADE_FACTOR = 0.6
COLOR_TO_MATPLOTLIB_COLOR = {
    "A": "red",
    "B": "blue",
    "C": "springgreen",
    "D": "fuchsia",
    "E": "darkorange",
    "F": "yellow",
    "G": "cyan",
}
MATPLOTLIB_COLOR_TO_SHADE_RANGE = {
    "yellow": (0.1, 0.7),
}


DIM_REDUCT_CLASS_TO_MODE = {PCA: "pca", UMAP: "umap"}
if HAS_CUML:
    DIM_REDUCT_CLASS_TO_MODE[cuml_UMAP] = "umap"


def plt_shade_metas_layerwise(metas: list[StandardTransformerMeta], sort_by_ids: bool = True):
    # Get all color types in metas
    colors = set(get_attr_list(metas, "vis_color"))

    # Create a dict mapping colors to lists of ordered metas with that color as vis_color
    if sort_by_ids:
        sorted_metas = sort_metas_by_id(metas)
    else:
        sorted_metas = metas

    color_metas_dict = {c: filter_metas(sorted_metas, layer_filter=lambda meta: meta.vis_color == c) for c in colors}

    # For each color and the (sorted) list of metas with that color
    # Assign each meta in the list of sorted metas an interpolated value between the darkest and lightest shade of that color based on its position in the sorted list
    # Because the metas are objects, the assigned values are accessible in the original metas list
    for c, c_metas in color_metas_dict.items():
        if c in COLOR_TO_MATPLOTLIB_COLOR:
            mpltlb_c = COLOR_TO_MATPLOTLIB_COLOR[c]
        else:
            try:
                mcolor.to_rgb(c)
                mpltlb_c = c
            except Exception as e:
                print(
                    f"Error occured while trying to map vis_color {c} to a color via mcolor.to_rgb(c). vis_color should either be one of the following keys OR a valid 'c' argument accepted by mcolor.to_rgb(c). But in this case it was neither. Valid keys: {COLOR_TO_MATPLOTLIB_COLOR.keys()}"
                )
                raise e

        if mpltlb_c in MATPLOTLIB_COLOR_TO_SHADE_RANGE:
            dark_shade_factor, light_shade_factor = MATPLOTLIB_COLOR_TO_SHADE_RANGE[mpltlb_c]
        else:
            dark_shade_factor, light_shade_factor = DARK_SHADE_FACTOR, LIGHT_SHADE_FACTOR

        # Define darkest and lightest color
        c_rgb = np.array(mcolor.to_rgb(mpltlb_c))
        c_rgb_darkest = np.clip(c_rgb * (1 - dark_shade_factor), 0, 1)
        white = np.array([1.0, 1.0, 1.0])
        c_rgb_lightest = np.clip(c_rgb + (white - c_rgb) * light_shade_factor, 0, 1)

        # Give each meta in c_metas a color which interpolates between c_rgb_darkest and c_rgb_lightest based on its position within c_metas
        normalized_positions = np.linspace(0, 1, len(c_metas))
        for meta, normalized_pos in zip(c_metas, normalized_positions):
            meta._matplotlib_color = (1 - normalized_pos) * c_rgb_darkest + normalized_pos * c_rgb_lightest


def vis_scatter(
    x: ArrayLike,
    y: ArrayLike,
    dims: tuple[int, int],
    c: ArrayLike | ColorType | Sequence[ColorType] | None,
    cmap: str | mcolor.Colormap | None,
    figsize: tuple[int, int] | None,
    dpi: int | None,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    title: str | None,
    s: float | ArrayLike,
    fontsize: float | int | str,
    labelsize: float | int | str,
    subplots_adjust_kwargs: dict[str, float | None] | None,
    label_prepend: str | None,
    disable_labels: bool,
    disable_ticks: bool,
    yaxis_formatter: plt.FuncFormatter | str | Callable | None,
    show_image: bool,
    return_image: bool,
) -> tuple[tuple[float, float], tuple[float, float], Image.Image | None]:
    """
    Create a scatter plot visualization of the dimensionally reduced data.

    Args:
        x (ArrayLike): X-coordinates for the scatter plot points
        y (ArrayLike): Y-coordinates for the scatter plot points
        dims (tuple[int, int]): The dimension indices being visualized (for axis labels)
        c (ArrayLike | ColorType | Sequence[ColorType] | None): Argument for c argument of ax.scatter(...)
        cmap (str | mcolor.Colormap | None): Argument for cmap argument of ax.scatter(...)
        figsize (tuple[int, int] | None): Figure size as (width, height) in inches
        dpi (int | None): Figure resolution in dots per inch
        xlim (tuple[float, float] | None): X-axis limits as (min, max), or None for auto-scaling
        ylim (tuple[float, float] | None): Y-axis limits as (min, max), or None for auto-scaling
        title (str | None): Plot title
        s (float | ArrayLike): Marker size for scatter plot points
        fontsize (float | int | str): Font size for axis labels and title
        labelsize (float | int | str): Font size for tick marks
        subplots_adjust_kwargs (dict[str, float | None] | None): kwargs for fig.subplots_adjust(...). If None, then fig.subplots_adjust will not be called
        label_prepend (str | None): A string prepended to the x labels and y labels if disable_labels is False
        disable_labels (bool): Disables x and y axis labels
        disable_ticks (bool): Disables tick marks and tick labels on both axes
        yaxis_formatter (plt.FuncFormatter | str | Callable | None): Function to format y-axis labels. If None, then y-axis labels will not be formatted.
        show_image (bool): Whether to show the plot
        return_image (bool): If True, return the plot as a PIL Image object

    Returns:
        tuple: A tuple containing:
            - xlim (tuple[float, float]): The actual x-axis limits used
            - ylim (tuple[float, float]): The actual y-axis limits used
            - image (PIL.Image | None): The plot as a PIL Image if return_image is True, otherwise None
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if subplots_adjust_kwargs is not None:
        fig.subplots_adjust(**subplots_adjust_kwargs)

    ax.scatter(x, y, c=c, s=s, cmap=cmap)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if yaxis_formatter is not None:
        ax.yaxis.set_major_formatter(yaxis_formatter)

    # Add labels and title
    if not disable_labels:
        if label_prepend is None:
            label_prepend = ""
        ax.set_xlabel(label_prepend + f"Dim {dims[0]}", fontsize=fontsize)
        ax.set_ylabel(label_prepend + f"Dim {dims[1]}", fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # Set tick labgelsize or disable ticks entirely
    if disable_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(axis="both", which="major", labelsize=labelsize)

    # Show the plot
    if show_image:
        plt.show(block=False)

    image = None
    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        image.load()
        buf.close()

    plt.close(fig)

    return xlim, ylim, image


def vis_2d_dim_reduct(
    lsdata: LSData, config: Vis2DDimReductConfig | None = None, prev_dim_reduct: PCA | UMAP | None = None
) -> dict:
    """
    Visualize high-dimensional latent state data in 2D using dimensionality reduction techniques.

    This function takes LSData containing high-dimensional representations and projects them into 2D space
    for visualization using either PCA or UMAP. The function supports various data preprocessing options,
    averaging strategies, and visualization customizations.

    Args:
        lsdata (LSData): The latent state data to visualize. Must contain data tensor and metadata with vis_color attributes.
        config (Vis2DDimReductConfig | None): Configuration for visualization. If None, defaults are used.
        prev_dim_reduct (PCA | UMAP | None): A pre-fitted dimensionality reduction model to use. If None, a new model will be fitted.

    Returns:
        A dictionary containing the fitted dim reduct object, the limits used on the figures, and if return_image is True, the image(s)

    Raises:
        ValueError: If both dim_reduct_mode and prev_dim_reduct are None, or if dim_reduct_mode is not
            a valid option, or if prev_dim_reduct is not a valid dimensionality reduction model,
            or if metadata lacks vis_color attributes, or if vis_color values are invalid.

    Note:
        - All metadata objects in lsdata must have a vis_color attribute with values from COLOR_TO_MATPLOTLIB_COLOR.
        - When using mean_colors=True, data points with the same vis_color will be averaged together.
        - The disable_pca_transform_centering parameter only affects PCA and will be ignored for UMAP.
        - If both dim_reduct_mode and prev_dim_reduct are provided, they must be compatible types.
    """

    """ Argument processing and checking """

    # One of dim_reduct_mode or prev_dim_reduct must be specified
    # If both are specified, they must match

    if config is None:
        config = Vis2DDimReductConfig()

    # attributes that may get mutated later, so we make local variable
    dim_reduct_mode = config.dim_reduct_mode
    n_components = config.n_components
    dim_reduct_kwargs = config.dim_reduct_kwargs
    n_fit = config.n_fit
    n_transform = config.n_transform
    vis_dims = config.vis_dims
    title = config.title
    sequence_cmap = config.sequence_cmap
    gpu_accelerate = config.gpu_accelerate

    if dim_reduct_mode is None and prev_dim_reduct is None:
        raise ValueError("dim_reduct_mode and prev_dim_reduct cannot both be None")

    if dim_reduct_mode is not None and dim_reduct_mode not in DIM_REDUCT_CLASS_TO_MODE.values():
        raise ValueError(
            f"dim_reduct_mode must be one of {set(DIM_REDUCT_CLASS_TO_MODE.values())} or None if prev_dim_reduct is specified. Instead got {dim_reduct_mode}"
        )

    if prev_dim_reduct is not None and not isinstance(prev_dim_reduct, tuple(DIM_REDUCT_CLASS_TO_MODE)):
        raise ValueError(
            f"prev_dim_reduct must be an instance of one of {DIM_REDUCT_CLASS_TO_MODE.keys()} or None if dim_reduct_mode is specified. Instead got {type(prev_dim_reduct)}"
        )

    if (
        dim_reduct_mode is not None
        and prev_dim_reduct is not None
        and dim_reduct_mode != DIM_REDUCT_CLASS_TO_MODE[type(prev_dim_reduct)]
    ):
        warnings.warn(
            f"dim_reduct_mode and prev_dim_reduct must match if both are specified. Instead got dim_reduct_mode: {dim_reduct_mode} and prev_dim_reduct: {type(prev_dim_reduct)}. dim_reduct_mode will be changed to {DIM_REDUCT_CLASS_TO_MODE[type(prev_dim_reduct)]}"
        )
        dim_reduct_mode = DIM_REDUCT_CLASS_TO_MODE[type(prev_dim_reduct)]

    if dim_reduct_mode is None:
        dim_reduct_mode = DIM_REDUCT_CLASS_TO_MODE[type(prev_dim_reduct)]

    if title is not None:
        if len(title) > 0:
            title += " "
        title += f"[{dim_reduct_mode}]"

    dim_reduct = prev_dim_reduct

    if not config.mean_layers:
        for meta in lsdata.metas:
            if not hasattr(meta, "vis_color"):
                raise ValueError("Not all metas in lsdata had a vis_color attribute!")

    if gpu_accelerate:
        if dim_reduct_mode == "umap":
            if not HAS_CUML:
                warnings.warn(
                    "gpu_accelerate was True and dim_reduct_mode was 'umap', but cuml package is not available. Please install it. Using non-paralleized UMAP"
                )
                gpu_accelerate = False
            if not torch.cuda.is_available():
                warnings.warn(
                    "gpu_accelerate was True and dim_reduct_mode was 'umap', but torch.cuda.is_available() was False. It must be True to use gpu_accelerate. Using non-accelerated UMAP"
                )
                gpu_accelerate = False

    if dim_reduct_kwargs and "n_components" in dim_reduct_kwargs:
        dim_reduct_kwargs = dim_reduct_kwargs.copy()
        if n_components is None:
            n_components = dim_reduct_kwargs.pop("n_components")
        elif n_components == dim_reduct_kwargs["n_components"]:
            del dim_reduct_kwargs["n_components"]
        else:
            raise ValueError(
                f"n_components was specified as an argument AND it was a key of dim_reduct_kwargs, yet they had differing values ({n_components} vs {dim_reduct_kwargs['n_components']}). Please make sure they are the same or specify only one"
            )

    if n_fit is not None and dim_reduct_mode == "pca":
        warnings.warn("Are you sure you want to fit PCA to only a subset of data? PCA is usually very fast...")

    rng = np.random.default_rng(config.seed)

    if gpu_accelerate:
        _UMAP = cuml_UMAP
    else:
        _UMAP = UMAP

    """ Any data processing that affects data tensor(s) shape """

    data = lsdata.data
    metas = lsdata.metas

    if config.mean_colors:
        colors = set(get_attr_list(metas, "vis_color"))
        meaned_data = []
        meaned_metas = []

        for c in colors:
            # Indices of metas with this vis_color
            c_idxs = [i for i, meta in enumerate(metas) if meta.vis_color == c]
            # Layer data corresponding to layers with this vis_color
            c_data = data[c_idxs]
            # Mean c_data across layer dim
            meaned_data.append(c_data.mean(axis=0, keepdims=True))

            # New meta to represent this new meaned color layer
            # It only contains vis_color info, because all other attributes are now undefined since it can represent the mean of many different layers
            new_meta = Metadata(id=None)
            new_meta.vis_color = c
            meaned_metas.append(new_meta)

        meaned_data = np.concatenate(meaned_data, axis=0)

        data = meaned_data
        metas = meaned_metas

    if config.mean_layers:
        data = data.mean(axis=0, keepdims=True)
        new_meta = Metadata(id=None)
        new_meta.vis_color = "black"
        metas = [new_meta]

    if config.mean_samples:
        data = data.mean(axis=1, keepdims=True)

    if config.mean_seq:
        data = data.mean(axis=2, keepdims=True)

    n_layers, n_samples, sequence_length, n_embed = data.shape
    n_original_data_points = n_layers * n_samples * sequence_length

    """ Processing that does not affect data shape """

    if config.convert_layer_norm:
        data = apply_layer_norm(data)

    if config.convert_unit_vector:
        data = convert_unit_vectors(data)

    """ Do dim reduct fit """

    # Create and fit dim_reduct if it was not provided in prev_dim_reduct

    if dim_reduct is None:
        # flatten data into 2d array, list of latent states
        fit_data = data.reshape((-1, n_embed))

        if n_components is None:
            n_components = 2

        # In case its None
        dim_reduct_kwargs = dim_reduct_kwargs or {}

        if n_fit is not None:
            # make n_fit a count
            if isinstance(n_fit, float):
                if n_fit <= 1:
                    n_fit *= len(fit_data)
                n_fit = round(n_fit)
            fit_data = rng.choice(fit_data, size=n_fit, replace=False)
            if config.verbose:
                print(
                    f"Fitting dim reduct with {n_fit} random data points. That's {100 * n_fit / n_original_data_points:.3f}% of the data!"
                )
        else:
            if config.verbose:
                print(f"Fitting dim reduct with full complement of {n_original_data_points} data points.")

        with timer(message="Dim reduct fit complete! Time taken: {}", disable=not config.verbose):
            if dim_reduct_mode == "pca":
                dim_reduct = PCA(n_components=n_components, random_state=config.seed, **dim_reduct_kwargs)
                dim_reduct.fit(fit_data)
            elif dim_reduct_mode == "umap":
                dim_reduct = _UMAP(n_components=n_components, random_state=config.seed, **dim_reduct_kwargs)
                dim_reduct.fit(fit_data)

    else:
        if n_components is not None:
            warnings.warn(
                f"n_components was specified ({n_components}) but prev_dim_reduct was also specified, so prev_dim_reduct will be used with however many components it was originally set with, and n_components will be ignored"
            )
        if dim_reduct_kwargs is not None:
            warnings.warn(
                f"dim_reduct_kwargs was specified ({n_components}) but prev_dim_reduct was also specified, so dim_reduct_kwargs will not be utilized as it is used only when creating a new dim reduct"
            )

    """ Do dim reduct transform on data for visualization """

    if n_transform is not None:
        # make n_transform a proportion
        if n_transform > 1:
            n_transform = n_transform / n_original_data_points

        n_per_layer = round(n_transform * n_samples * sequence_length)

        transform_data = []
        for i in range(n_layers):
            layer_data = data[i].reshape(-1, n_embed)
            transform_data.append(rng.choice(layer_data, size=n_per_layer, replace=False, shuffle=False))

        transform_data = np.concatenate(transform_data, axis=0)

        if config.verbose:
            print(
                f"Transforming/visualizng dim reduct with {n_per_layer} random data points per layer for {n_layers} layers: Total {n_per_layer * n_layers} data points. That's {100 * n_layers * n_per_layer / n_original_data_points:.3f}% of the data!"
            )
    else:
        if config.verbose:
            print(f"Transforming/visualizng dim reduct with full complement of {n_original_data_points} data points.")
        transform_data = data.reshape((-1, n_embed))
        n_per_layer = n_samples * sequence_length

    # Perform the transform with dim_reduct
    umap_types = (UMAP,)
    if HAS_CUML:
        umap_types += (cuml_UMAP,)

    with timer(message="Dim reduct transform complete! Time taken: {}", disable=not config.verbose):
        if isinstance(dim_reduct, PCA):
            if config.disable_pca_transform_centering:
                vis_data = transform_data @ dim_reduct.components_.T
            else:
                vis_data = dim_reduct.transform(transform_data)
        elif isinstance(dim_reduct, umap_types):
            vis_data = dim_reduct.transform(transform_data)

    """ Visualize """

    ## Shade colors based on layer

    plt_shade_metas_layerwise(metas, sort_by_ids=not config.mean_colors)

    # Create list to define the colors of each point based on its layer

    if not config.mean_layers or sequence_cmap is None or n_transform is not None:
        c_list = []
        for c in get_attr_list(metas, "_matplotlib_color"):
            c_list += [c] * n_per_layer
        sequence_cmap = None
    else:
        c_list = [i for i in range(sequence_length) for _ in range(n_samples)]
        if config.reverse_sequence_cmap:
            c_list.reverse()

    # See if vis_dims is actually a collection of dims to visualize
    # This check is not perfect but good enough.
    if vis_dims is None:  # Default
        vis_dims = (0, 1)

    is_many_dim_vis = hasattr(vis_dims[0], "__getitem__")

    if not is_many_dim_vis:
        vis_dims = [vis_dims]

    images = []

    new_vis_lims = {}

    for vis_dim in vis_dims:
        x = vis_data[:, vis_dim[0]]
        y = vis_data[:, vis_dim[1]]

        vis_title = title
        if is_many_dim_vis and title is not None:
            vis_title += f" [ Dims ({vis_dim[0]},{vis_dim[1]}) ]"

        if config.vis_lims is not None:
            if tuple(vis_dim) in config.vis_lims:
                xlim, ylim = config.vis_lims[tuple(vis_dim)]
            elif (flipped_vis_dim := (vis_dim[1], vis_dim[0])) in config.vis_lims:
                ylim, xlim = config.vis_lims[flipped_vis_dim]
            else:
                warnings.warn(
                    f"vis_lims was not None but the dimensions {vis_dim} were not found in vis_lims. Thus the xlim and ylim values will be automatically determined"
                )
                xlim, ylim = None, None
        else:
            xlim, ylim = None, None

        with timer(message=f"Visualization for dims {vis_dim} complete. Time taken: {{}}", disable=not config.verbose):
            if config.verbose:
                print("Visualizing scatter plot")
                if not config.show_image:
                    print("No image will be displayed because show_image is False")
            xlim, ylim, image = vis_scatter(
                x=x,
                y=y,
                dims=vis_dim,
                c=c_list,
                cmap=sequence_cmap,
                figsize=config.figsize,
                dpi=config.dpi,
                xlim=xlim,
                ylim=ylim,
                title=vis_title,
                s=config.s,
                fontsize=config.fontsize,
                labelsize=config.labelsize,
                subplots_adjust_kwargs=config.subplots_adjust_kwargs,
                label_prepend=dim_reduct_mode.upper() + " ",
                disable_labels=config.disable_labels,
                disable_ticks=config.disable_ticks,
                yaxis_formatter=config.yaxis_formatter,
                show_image=config.show_image,
                return_image=True,
            )

        new_vis_lims[tuple(vis_dim)] = xlim, ylim

        images.append(image)

    vis_results = {"dim_reduct": dim_reduct, "vis_lims": new_vis_lims, "images": images}

    return vis_results


def plot_values_bar(
    values: ArrayLike,
    color: Any = None,  # color or list of colors
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    fontsize: float | int | str = 10,
    labelsize: float | int | str = 10,
    figsize: ArrayLike | None = None,
    dpi: int | None = None,
    x_indices: ArrayLike = None,
    width: float = 0.8,
    subplots_adjust_kwargs: dict[str, float | None] | None = None,
    yaxis_formatter: plt.FuncFormatter | str | Callable | None = None,
    disable_ticks: bool = False,
    disable_x_ticks: bool = False,
    disable_y_ticks: bool = False,
    log_y: bool = False,
    return_image: bool = False,
    show_image: bool = True,
):
    """
    Create a bar graph of values. Colors provided via color

    Args:
        values (ArrayLike): An ArrayLike sequence containing one values for the bar plot.
        color (Any): Color or list of colors as allowed by Matplotlib. Directly passed to plt.bar
        xlabel (str | None): xlabel argument for plt
        ylabel (str | None): ylabel argument for plt
        title (str | None): title argument for plt
        fontsize (float | int | str): Font size for axis labels and title
        labelsize (float | int | str): Font size for tick marks
        figsize (str | None): figsize argument for plt.figure
        dpi (int | None): dpi argument for plt.figure
        x_indices (ArrayLike): x_indices to use for bar plot. If None, will be auto-generated starting at 0
        width (float): Width of each bar. Defaults to 0.8. 1.0 means no gap between bars
        subplots_adjust_kwargs (dict[str, float | None] | None): kwargs for fig.subplots_adjust(...). If None, then fig.subplots_adjust will not be called
        yaxis_formatter (plt.FuncFormatter | str | Callable | None): Function to format y-axis labels. If None, then y-axis labels will not be formatted.
        return_image (bool): Whether to return the image as an Image.Image. If None will default to False
        disable_ticks (bool): Disables tick marks and tick labels on both axes
        disable_x_ticks (bool): Disables tick marks and tick labels on x-axis
        disable_y_ticks (bool): Disables tick marks and tick labels on y-axis
        show_image (bool): Whether to show the image
        log_y (bool): Whether to use a log scale for the y-axis
        return_image (bool): Whether to return the image as an Image.Image
        show_image (bool): Whether to show the image


    Returns:
        Image: Bar graph image if return_image argument was True
    """

    if x_indices is None:
        # Generate indices for the x-axis
        x_indices = list(range(len(values)))
    if return_image is None:
        return_image = False

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if subplots_adjust_kwargs is not None:
        fig.subplots_adjust(**subplots_adjust_kwargs)

    if yaxis_formatter is not None:
        ax.yaxis.set_major_formatter(yaxis_formatter)

    if log_y:
        ax.set_yscale("log")

    if disable_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    elif disable_x_ticks:
        ax.set_xticks([])
    elif disable_y_ticks:
        ax.set_yticks([])
    elif labelsize is not None:
        ax.tick_params(axis="both", which="major", labelsize=labelsize)

    # Bar plot
    ax.bar(x_indices, values, color=color, width=width)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        if log_y:
            ylabel += " (log)"
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    if show_image:
        plt.show(block=False)

    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        image.load()
        buf.close()

    plt.close(fig)

    if return_image:
        return image


def plot_values_by_layer(
    layer_values: ArrayLike,
    metas: Sequence[StandardTransformerMeta],
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: ArrayLike | None = None,
    dpi: int | None = None,
    return_image: bool = None,
    **kwargs,
):
    """
    Create a bar graph of layer_values. Colors provided via the vis_color attribute in each meta in metas.

    Args:
        layer_values (ArrayLike): An ArrayLike sequence containing one value for every single layer in metas.
        metas (Sequence[StandardTransformerMeta]): A sequence of StandardTransformerMeta. vis_color must be defined for each meta
        xlabel (str | None): xlabel argument for plt
        ylabel (str | None): ylabel argument for plt
        title (str | None): title argument for plt
        figsize (str | None): figsize argument for plt.figure
        dpi (int | None): dpi argument for plt.figure
        return_image (bool): Whether to return the image as an Image.Image. If None will default to False
    """

    assert len(layer_values) == len(metas), (
        f"the length of layer_values must equal the length of metas, since each value in layer_values should correspond to one layer described by metas. But the lengths were not equal! Got len(layer_values): {len(layer_values)} but len(metas): {len(metas)}"
    )

    # Alternate colors: blue, red, blue, ...
    plt_shade_metas_layerwise(metas)
    colors = [meta._matplotlib_color for meta in metas]

    return plot_values_bar(
        values=layer_values,
        color=colors,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
        dpi=dpi,
        return_image=return_image,
        **kwargs,
    )


def plot_vector_norm_by_layer(
    data: LSData,
    xlabel: str | None = "Layers",
    ylabel: str | None = "Mean Norm",
    title: str | None = "Mean Norm vs Layers",
    figsize: ArrayLike | None = None,
    dpi: int | None = None,
    return_image: bool = None,
    **kwargs,
):
    """
    Create a bar graph of the mean vector norm (length) of each layer's hidden states.
    Each meta must have a vis_color defined.

    Args:
        data (LSData): LSData object
        xlabel (str | None): xlabel argument for plt
        ylabel (str | None): ylabel argument for plt
        title (str | None): title argument for plt
        figsize (str | None): figsize argument for plt.figure
        dpi (int | None): dpi argument for plt.figure
        return_image (bool): Whether to return the image as an Image.Image. If None will default to False
    """
    # Calculate norm of the hidden states
    ls_l2 = np.linalg.norm(data.data, axis=-1)

    # Mean across all but the 0th dim (layer dim)
    dims_to_reduce = tuple(range(1, ls_l2.ndim))
    mean_lengths = ls_l2.mean(axis=dims_to_reduce)

    assert mean_lengths.ndim == 1, (
        f"Expected mean_lengths to have 1 dim but instead it has {mean_lengths.ndim} dims! Something has gone wrong"
    )

    return plot_values_by_layer(
        mean_lengths,
        metas=data.metas,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
        dpi=dpi,
        return_image=return_image,
        **kwargs,
    )


def plot_vector_norm_by_axis(
    data: np.ndarray | LSData,
    axis: int,
    color: Any = None,
    xlabel: str | None = "Axis",
    ylabel: str | None = "Mean Norm",
    title: str | None = "Mean Norm vs Axis",
    figsize: ArrayLike | None = None,
    dpi: int | None = None,
    x_indices_start: int | None = None,
    return_image: bool = None,
    **kwargs,
):
    """
    Create a bar graph of the mean vector norm (length) along a certain axis.

    Args:
        data (ndarray | LSData): Numpy array or LSData object
        axis (int): The axis along which to graph the values. Must be a non-negative integer
        color (Any): Color or list of colors as allowed by Matplotlib. Directly passed to plt.bar
        xlabel (str | None): xlabel argument for plt
        ylabel (str | None): ylabel argument for plt
        title (str | None): title argument for plt
        figsize (str | None): figsize argument for plt.figure
        dpi (int | None): dpi argument for plt.figure
        x_indices_start (int | None): Starting value for the x-indices on the bar plot. If None defaults to 0
        return_image (bool): Whether to return the image as an Image.Image. If None will default to False
    """

    if x_indices_start is None:
        x_indices_start = 0

    if isinstance(data, LSData):
        data = data.data

    # Calculate norm of the hidden states
    ls_l2 = np.linalg.norm(data, axis=-1)

    # Mean across all but the specified axis
    dims_to_reduce = list(range(0, ls_l2.ndim))
    dims_to_reduce.remove(axis)
    dims_to_reduce = tuple(dims_to_reduce)
    mean_lengths = ls_l2.mean(axis=dims_to_reduce)

    assert mean_lengths.ndim == 1, (
        f"Expected mean_lengths to have 1 dim but instead it has {mean_lengths.ndim} dims! Something has gone wrong"
    )

    x_indices = list(range(x_indices_start, x_indices_start + len(mean_lengths)))

    return plot_values_bar(
        values=mean_lengths,
        color=color,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        figsize=figsize,
        dpi=dpi,
        x_indices=x_indices,
        return_image=return_image,
        **kwargs,
    )


def stitch_images_grid(
    images: Sequence[Image.Image],
    n_rows: int,
    n_columns: int,
    padding_color="white",
    image_padding: int | tuple[int, int] = 0,
) -> Image.Image:
    """Validation & Processing"""

    if isinstance(image_padding, int):
        image_padding_width = image_padding
        image_padding_height = image_padding
    else:
        image_padding_width, image_padding_height = image_padding

    n_images = len(images)

    if n_rows * n_columns < n_images:
        raise ValueError(
            f"Not enough rows/columns for all images. There were {n_images} images but only {n_rows} rows * {n_columns} columns = {n_rows * n_columns} grid spots"
        )

    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    if any(image.width != max_width for image in images):
        print(f"Not all images had the same width. They will all be padded to the max width ({max_width})")
    if any(image.height != max_height for image in images):
        print(f"Not all images had the same height. They will all be padded to the max height ({max_height})")

    """ Creating Stitched Image """

    padded_image_width = max_width + 2 * image_padding_width
    padded_image_height = max_height + 2 * image_padding_height

    # Padding
    images = [
        ImageOps.pad(image, size=(padded_image_width, padded_image_height), color=padding_color) for image in images
    ]

    canvas_width = n_columns * padded_image_width
    canvas_height = n_rows * padded_image_height

    canvas = Image.new("RGB", size=(canvas_width, canvas_height), color=padding_color)

    for idx, image in enumerate(images):
        row, col = divmod(idx, n_columns)

        x0 = col * padded_image_width
        y0 = row * padded_image_height

        canvas.paste(image, (x0, y0))

    return canvas


def visualizer(
    visualize_configs: VisualizeConfig | list[VisualizeConfig] | tuple[VisualizeConfig],
    vis_dim_reduct_configs: Vis2DDimReductConfig
    | None
    | list[Vis2DDimReductConfig | None]
    | tuple[Vis2DDimReductConfig | None] = None,
    return_visualized: bool = False,
):
    """Checks"""

    if isinstance(visualize_configs, (tuple, list)) and isinstance(vis_dim_reduct_configs, (tuple, list)):
        if len(visualize_configs) != len(vis_dim_reduct_configs):
            raise ValueError("The number of visualize_configs must equal the number of vis_dim_reduct_configs")
    elif isinstance(visualize_configs, (tuple, list)):
        vis_dim_reduct_configs = [vis_dim_reduct_configs] * len(visualize_configs)
    elif isinstance(vis_dim_reduct_configs, (tuple, list)):
        visualize_configs = [visualize_configs] * len(vis_dim_reduct_configs)
    else:
        visualize_configs = [visualize_configs]
        vis_dim_reduct_configs = [vis_dim_reduct_configs]

    """ Dim reduct """

    if return_visualized:
        visualized_return = []

    prev_vis = None

    for vis_config, vis_dim_reduct_config in zip(visualize_configs, vis_dim_reduct_configs):
        ## Get data
        data_dir_path = get_project_root() / "processed_data" / vis_config.data_name
        lsdata = LSData(
            dir_path=data_dir_path,
            layer_filter=vis_config.layer_filter,
            sample_selection=vis_config.sample_selection,
            sequence_selection=vis_config.sequence_selection,
            max_workers=vis_config.max_workers,
        )

        ## Colorize data
        for meta in lsdata.metas:
            vis_config.colorizer(meta)

        ## Prev vis logic
        use_prev_dim_reduct = False
        use_prev_lims = False
        set_as_prev_vis = False
        if vis_config.prev_vis_mode == 1:
            set_as_prev_vis = True
        elif vis_config.prev_vis_mode == 2:
            use_prev_dim_reduct = True
        elif vis_config.prev_vis_mode == 3:
            use_prev_dim_reduct = True
            use_prev_lims = True

        if (use_prev_dim_reduct or use_prev_lims) and prev_vis is None:
            warnings.warn(
                "use_prev_dim_reduct and/or use_prev_lims was True but prev_vis is None! Setting both use_prev_dim_reduct and use_prev_lims to False"
            )
            use_prev_dim_reduct = False
            use_prev_lims = False

        if prev_vis is not None and use_prev_lims and (prev_vis["vis_lims"] is None):
            warnings.warn(
                "use_prev_lims was specified, but prev_vis['vis_lims'] was None, so no previous lims will be used!"
            )

        ## Visualize

        if vis_dim_reduct_config is None:
            vis_dim_reduct_config = Vis2DDimReductConfig()

        prev_dim_reduct = prev_vis["dim_reduct"] if use_prev_dim_reduct else None

        if use_prev_lims:
            vis_dim_reduct_config = replace(vis_dim_reduct_config, vis_lims=prev_vis["vis_lims"])

        if vis_config.do_many_vis:
            if vis_dim_reduct_config.n_components is None:
                raise ValueError(
                    "vis_dim_reduct_config.n_components is None! This is not allowed when do_many_vis is True"
                )

            # A list of all (i,j) tuples where i < j
            vis_dims_list = list(zip(*np.triu_indices(vis_dim_reduct_config.n_components, k=1)))
            # Convert np ints to ints for printout aesthetics
            vis_dims_list = [(i.item(), j.item()) for i, j in vis_dims_list]

            vis_dim_reduct_config = replace(vis_dim_reduct_config, vis_dims=vis_dims_list)

        # TODO: Refactor so this doesn't change global, but only per-fig/ax
        if vis_config.font_family is not None:
            plt.rcParams["font.family"] = vis_config.font_family

        visualized = vis_2d_dim_reduct(lsdata=lsdata, config=vis_dim_reduct_config, prev_dim_reduct=prev_dim_reduct)

        if set_as_prev_vis:
            prev_vis = visualized

        save_dir = Path(vis_config.img_save_dir) if vis_config.img_save_dir is not None else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        ## Stich images (if applicable) + Save images (if specified)

        if vis_config.do_many_vis:
            stitched_image = stitch_images_grid(
                visualized["images"],
                n_rows=vis_config.composite_vis_grid_shape[0],
                n_columns=vis_config.composite_vis_grid_shape[1],
                image_padding=vis_config.composite_img_padding,
            )
            visualized["stitched_image"] = stitched_image
            if save_dir:
                stitched_image.save(save_dir / "stitched_img.png")
            if vis_dim_reduct_config.show_image:
                display(stitched_image)

        if save_dir:
            for i, image in enumerate(visualized["images"]):
                image.save(save_dir / f"img-{i}.png")

        if return_visualized:
            visualized_return.append(visualized)

    if return_visualized:
        return visualized_return
