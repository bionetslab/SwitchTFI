
import pandas as pd
import scanpy as sc

import importlib.resources as resources
# import pkg_resources
import lzma
import pickle


def _load_anndata(filename_base: str) -> sc.AnnData:

    # h5ad_path = pkg_resources.resource_filename(__name__, f'd/{filename_base}.h5ad')
    # compressed_path = pkg_resources.resource_filename(__name__, f'd/{filename_base}.pickle.xz')

    package = 'switchtfi'

    with (
        resources.as_file(resources.files(package) / f'd/{filename_base}.h5ad') as h5ad_path,
        resources.as_file(resources.files(package) / f'd/{filename_base}.pickle.xz') as compressed_path
    ):

        try:
            adata = sc.read_h5ad(h5ad_path)
        except FileNotFoundError:
            with lzma.open(compressed_path, 'rb') as f:
                adata = pickle.load(f)
            sc.write(h5ad_path, adata)

    return adata


def preendocrine_alpha() -> sc.AnnData:
    """
    Load the preprocessed pre-endocrine alpha scRNA-seq dataset.

    This function attempts to load the AnnData object from an ``.h5ad`` file. Upon its first use it loads the data from a compressed pickle file (``.pickle.xz``), then saves it as an ``.h5ad`` file for future use.

    Returns:
        sc.AnnData: The pre-endocrine alpha AnnData object.
    """
    return _load_anndata(filename_base='pre-endocrine_alpha')


def preendocrine_beta() -> sc.AnnData:
    """
    Load the preprocessed pre-endocrine beta scRNA-seq dataset.

    This function attempts to load the AnnData object from an ``.h5ad`` file. Upon its first use it loads the data from a compressed pickle file (``.pickle.xz``), then saves it as an ``.h5ad`` file for future use.

    Returns:
        sc.AnnData: The pre-endocrine beta AnnData object.
    """
    return _load_anndata(filename_base='pre-endocrine_beta')


def erythrocytes() -> sc.AnnData:
    """
    Load the preprocessed erythrocyte scRNA-seq dataset.

    This function attempts to load the AnnData object from an ``.h5ad`` file. Upon its first use it loads the data from a compressed pickle file (``.pickle.xz``), then saves it as an ``.h5ad`` file for future use.

    Returns:
        sc.AnnData: The erythrocyte AnnData object.
    """
    return _load_anndata(filename_base='erythrocytes')


def preendocrine_alpha_grn() -> pd.DataFrame:
    """
    Load the GRN inferred with Scenic for the pre-endocrine alpha-cell transition scRNA-seq data.

    Returns:
        pd.DataFrame: The GRN as an edge-list.
    """
    return pd.read_csv(
        pkg_resources.resource_filename(__name__, 'd/ngrnthresh9_alpha_pyscenic_combined_grn.csv'), index_col=0
    )


def preendocrine_beta_grn() -> pd.DataFrame:
    """
    Load the GRN inferred with Scenic for the pre-endocrine beta-cell transition scRNA-seq data.

    Returns:
        pd.DataFrame: The GRN as an edge-list.
    """
    return pd.read_csv(
        pkg_resources.resource_filename(__name__, 'd/ngrnthresh9_beta_pyscenic_combined_grn.csv'), index_col=0
    )


def erythrocytes_grn() -> pd.DataFrame:
    """
    Load the GRN inferred with Scenic for the erythrocyte differentiation scRNA-seq data.

    Returns:
        pd.DataFrame: The GRN as an edge-list.
    """
    return pd.read_csv(
        pkg_resources.resource_filename(__name__, 'd/ngrnthresh9_erythrocytes_pyscenic_combined_grn.csv'), index_col=0
    )
