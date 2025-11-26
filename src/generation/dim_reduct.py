import gc
import argparse
import copy
import shutil
import numpy as np
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

from src.lsdata.LatentStates import LSData, save_lsdata
from src.utils import get_project_root, save


# def dim_reduct(lsdata: LSData, new_dim: int):
#     """
#     Given a LSData object, performs PCA on all the data to new_dim and then returns a new LSData object with reduced dimensionality

#     Importantly, does NOT perform mean-centering during the PCA transform (PCA still uses mean-centering during fit). This is to preserve the relationship of the vectors to the origin
#     """

#     data = lsdata.data

#     old_shape = list(data.shape)

#     new_shape = copy.deepcopy(old_shape)
#     new_shape[-1] = new_dim

#     old_dim = old_shape[-1]
#     if new_dim >= old_dim:
#         raise ValueError(
#             f"The new dimensionality ({new_dim}) was greater than or equal to the existing dimensionality! ({old_dim})"
#         )

#     data = data.reshape(-1, old_dim)

#     pca = PCA(n_components=new_dim)

#     pca.fit(data)

#     # Transforms data without mean-centering
#     data = data @ pca.components_.T

#     data = data.reshape(new_shape)

#     return LSData(
#         data=data,
#         metas=copy.deepcopy(lsdata.metas),
#         token_ids=copy.deepcopy(lsdata.token_ids),
#         token_strings=copy.deepcopy(lsdata.token_strings),
#     )


def dim_reduct_from_name(data_name: str, new_dim: int, new_name_override: str | None = None, n_fit: int | None = None, n_fit_samples: int | None = None):
    """
    Given existing data, perform PCA to reduce the overall dimensionality and then save as a new dataset

    Args:
        data_name (str): The name of the existing processed data

        new_dim (int): The new dimensionality of the data

        new_name_override (str | None): Overrides the automatically generated name of the new data

        n_fit (int | None): If specified, will fit PCA on a random selection of n_fit latent states instead of all latent states

        n_fit_samples (int | None): If specified, will fit PCA on only the specified number of samples instead of using all samples.
    """

    """Resolve new data dir and check if already exists"""

    if new_name_override is None:
        new_name = data_name + f"-DIM_REDUCT_{new_dim}"
    else:
        new_name = new_name_override

    reduct_data_dir_path = get_project_root() / "processed_data" / new_name

    if reduct_data_dir_path.exists():
        raise ValueError(f"The path at '{reduct_data_dir_path}' already exists!")

    """ Load Data """
    ## Below here, we make many design choices here based on minimizing peak memory usage, because these hidden states can be massive...

    data_dir_path = get_project_root() / "processed_data" / data_name

    if not data_dir_path.exists():
        raise ValueError(f"The data directory {data_dir_path} does not exist!")

    if n_fit_samples is not None:
        layer_0_data = np.load(data_dir_path / "0.npy")
        n_samples = layer_0_data.shape[-3]
        del layer_0_data
        if n_fit_samples > n_samples:
            raise ValueError(f"n_fit_samples ({n_fit_samples}) greater than n_samples ({n_samples})!")

        rng = np.random.default_rng(42)
        sample_selection = rng.choice(n_samples, size=n_fit_samples, replace=False)
    else:
        sample_selection = None


    lsdata = LSData(dir_path=data_dir_path, sample_selection=sample_selection)

    data = lsdata.data
    metas = lsdata.metas

    del lsdata

    ''' Fit PCA '''

    old_shape = list(data.shape)

    new_shape = copy.deepcopy(old_shape)
    new_shape[-1] = new_dim

    old_dim = old_shape[-1]
    if new_dim >= old_dim:
        raise ValueError(
            f"The new dimensionality ({new_dim}) was greater than or equal to the existing dimensionality! ({old_dim})"
        )

    data = data.reshape(-1, old_dim)
    gc.collect()

    if n_fit is not None:
        rng = np.random.default_rng(42)
        data = rng.choice(data, size=n_fit, replace=False)
        gc.collect()

    print("Fitting PCA to data")

    pca = PCA(n_components=new_dim)

    pca.fit(data)
    del data
    gc.collect()

    print("PCA fit complete! Beginning dimensionality reduction/saving")

    ''' Dim reduct + Save '''
    
    reduct_data_dir_path.mkdir(parents=True, exist_ok=False)

    for i in tqdm(range(len(metas)), desc="Loading + dim reduce + saving"):
        layer_data = np.load(data_dir_path / f"{i}.npy")
        layer_data = layer_data @ pca.components_.T
        np.save(reduct_data_dir_path / f"{i}.npy", layer_data)
    
    shutil.copy(data_dir_path / "metadata.pkl", reduct_data_dir_path / "metadata.pkl")
    shutil.copy(data_dir_path / "token_ids.npy", reduct_data_dir_path / "token_ids.npy")
    shutil.copy(data_dir_path / "token_strings.npy", reduct_data_dir_path / "token_strings.npy")

    print(f"Completed dim reduct and saved new data to {reduct_data_dir_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_name", type=str)
    parser.add_argument("--new_dim", type=int)
    parser.add_argument("--new_name_override", type=str)
    parser.add_argument("--n_fit", type=int, default=None)
    parser.add_argument("--n_fit_samples", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dim_reduct_from_name(data_name=args.data_name, new_dim=args.new_dim, new_name_override=args.new_name_override, n_fit=args.n_fit, n_fit_samples=args.n_fit_samples)
