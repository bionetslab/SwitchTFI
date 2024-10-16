import pandas as pd
import scanpy as sc
import pkg_resources
import lzma
import pickle


def preendocrine_alpha() -> sc.AnnData:
    """
    Load the preprocessed pre-endocrine alpha scRNA-seq dataset.

    This function attempts to load the AnnData object from an `.h5ad` file.
    Upon its first use it loads the data from a compressed pickle file (`.pickle.xz`),
    then saves it as an `.h5ad` file for future use.

    Returns:
        sc.AnnData: The pre-endocrine alpha AnnData object.
    """
    try:
        ad = sc.read_h5ad(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_alpha.h5ad'))
    except FileNotFoundError:
        # Deserialize (unpickle) and decompress the AnnData object
        with lzma.open(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_alpha.pickle.xz'), 'rb') as f:
            ad = pickle.load(f)
        sc.write(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_alpha.h5ad'), ad)
    return ad


def preendocrine_beta() -> sc.AnnData:
    """
    Load the preprocessed pre-endocrine beta scRNA-seq dataset.

    This function attempts to load the AnnData object from an `.h5ad` file.
    Upon its first use it loads the data from a compressed pickle file (`.pickle.xz`),
    then saves it as an `.h5ad` file for future use.

    Returns:
        sc.AnnData: The pre-endocrine alpha AnnData object.
    """
    try:
        ad = sc.read_h5ad(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_beta.h5ad'))
    except FileNotFoundError:
        # Deserialize (unpickle) and decompress the AnnData object
        with lzma.open(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_beta.pickle.xz'), 'rb') as f:
            ad = pickle.load(f)
        sc.write(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_beta.h5ad'), ad)
    return ad


def erythrocytes() -> sc.AnnData:
    """
    Load the preprocessed erythrocyte scRNA-seq dataset.

    This function attempts to load the AnnData object from an `.h5ad` file.
    Upon its first use it loads the data from a compressed pickle file (`.pickle.xz`),
    then saves it as an `.h5ad` file for future use.

    Returns:
        sc.AnnData: The pre-endocrine alpha AnnData object.
    """
    try:
        ad = sc.read_h5ad(pkg_resources.resource_filename(__name__, '/d/erythrocytes.h5ad'))
    except FileNotFoundError:
        # Deserialize (unpickle) and decompress the AnnData object
        with lzma.open(pkg_resources.resource_filename(__name__, '/d/erythrocytes.pickle.xz'), 'rb') as f:
            ad = pickle.load(f)
        sc.write(pkg_resources.resource_filename(__name__, '/d/erythrocytes.h5ad'), ad)
    return ad


def preendocrine_alpha_grn() -> pd.DataFrame:
    """
    Load the GRN inferred with Scenic for the pre-endocrine alpha-cell transition scRNA-seq data.

    Returns:
        pd.DataFrame: The GRN as an edge-list.
    """
    return pd.read_csv(
        pkg_resources.resource_filename(__name__, '/d/ngrnthresh9_alpha_pyscenic_combined_grn.csv'), index_col=0
    )


def preendocrine_beta_grn() -> pd.DataFrame:
    """
    Load the GRN inferred with Scenic for the pre-endocrine beta-cell transition scRNA-seq data.

    Returns:
        pd.DataFrame: The GRN as an edge-list.
    """
    return pd.read_csv(
        pkg_resources.resource_filename(__name__, '/d/ngrnthresh9_beta_pyscenic_combined_grn.csv'), index_col=0
    )


def erythrocytes_grn() -> pd.DataFrame:
    """
    Load the GRN inferred with Scenic for the erythrocyte differentiation scRNA-seq data.

    Returns:
        pd.DataFrame: The GRN as an edge-list.
    """
    return pd.read_csv(
        pkg_resources.resource_filename(__name__, '/d/ngrnthresh9_erythrocytes_pyscenic_combined_grn.csv'), index_col=0
    )
