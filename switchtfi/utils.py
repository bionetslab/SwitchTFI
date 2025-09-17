
import scanpy as sc
import numpy as np
import scipy as sci
import pandas as pd

from typing import *


def csr_to_numpy(x: Union[np.ndarray, sci.sparse.csr_matrix]) -> np.ndarray:
    if isinstance(x, sci.sparse.csr_matrix):
        x = x.toarray()
    return x


def anndata_to_numpy(adata: sc.AnnData,
                     layer_key: Union[str, None] = None) -> np.ndarray:
    if layer_key is None:
        x = adata.X
        x = csr_to_numpy(x)
    else:
        x = adata.layers[layer_key]
        x = csr_to_numpy(x)

    return x


def load_grn_json(grn_path: str) -> pd.DataFrame:
    grn = pd.read_json(grn_path, precise_float=True)
    for c in grn.columns:
        # Get entries of dataframe that are not None
        not_nan_entries = grn[c][np.logical_not(grn[c].isna().to_numpy())].to_list()
        # Pick 1st entry and check dtype
        try:
            if isinstance(not_nan_entries[0], list):
                # Turn column of lists into column of numpy arrays
                grn[c] = grn[c].apply(np.array)
        except KeyError:  # Case: All column entries are None
            pass
    return grn


def labels_to_bool(clustering: np.ndarray) -> np.ndarray:
    cluster_labels = np.unique(clustering)

    if cluster_labels.shape[0] > 2:
        raise ValueError('Clustering must consist of at most 2 clusters.')

    cluster_bool = (clustering == cluster_labels[0])

    return cluster_bool


def solve_lsap(clust1: np.ndarray, clust2: np.ndarray) -> float:

    # Solve (trivial, only 2 possible cases) LSAP problem => Similarity score for the 2 clusterings
    # Case 1: L-C1, R-C2
    case1_ji1 = calc_ji(a=clust1, b=clust2)
    case1_ji2 = calc_ji(a=np.logical_not(clust1), b=np.logical_not(clust2))
    case1_obj_val = case1_ji1 + case1_ji2

    # Case 2: L-C2, R-C1
    case2_ji1 = calc_ji(a=clust1, b=np.logical_not(clust2))
    case2_ji2 = calc_ji(a=np.logical_not(clust1), b=clust2)
    case2_obj_val = case2_ji1 + case2_ji2

    weight = max(case1_obj_val, case2_obj_val) / 2

    return weight


def calc_ji(a: np.ndarray, b: np.ndarray) -> float:

    # JI(A, B) = #(A \cap B) / #(A \cup B) \in [0,1]
    # True entries in vector indicate that cell is contained in set
    if not a.dtype == 'bool' or not b.dtype == 'bool':
        raise ValueError('Input must be bool arrays.')

    if not a.shape[0] == b.shape[0]:
        raise ValueError('Input arrays must be of same dimension')

    if a.sum() == 0 and b.sum() == 0:
        ji = 0
    else:
        ji = (a * b).sum() / (a + b).sum()

    return ji


def get_regulons_old(grn: pd.DataFrame,
                 gene_names: Union[List[str], None] = None,
                 additional_info_keys: Union[List[str], None] = None,
                 tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> Union[Dict[str, List[str]], Dict[str, Dict]]:

    # If no gene names are passed, compute regulons of all TFs
    if gene_names is None:
        gene_names = np.unique(grn[tf_target_keys[0]].to_numpy()).tolist()

    if additional_info_keys is None:
        regulon_dict = {}
        for gene in gene_names:
            gene_tf_bool = (grn[tf_target_keys[0]].to_numpy() == gene)

            if gene_tf_bool.sum() == 0:
                print(f'WARNING: Gene {gene} is not a TF in the given GRN')

            targets = grn[tf_target_keys[1]].to_numpy()[gene_tf_bool].tolist()
            regulon_dict[gene] = targets

    else:
        regulon_dict = {}
        for gene in gene_names:
            gene_tf_bool = (grn[tf_target_keys[0]].to_numpy() == gene)

            if gene_tf_bool.sum() == 0:
                print(f'WARNING: Gene {gene} is not a TF in the given GRN')
            dummy = {'targets': grn[tf_target_keys[1]].to_numpy()[gene_tf_bool].tolist()}
            for key in additional_info_keys:
                dummy[key] = grn[key].to_numpy()[gene_tf_bool].tolist()

            regulon_dict[gene] = dummy

    return regulon_dict


def get_regulons(
        grn: pd.DataFrame,
        gene_names: Union[List[str], None] = None,
        additional_info_keys: Union[List[str], None] = None,
        tf_target_keys: Tuple[str, str] = ('TF', 'target')
) -> Dict[str, Dict]:

    # If no gene names are passed, compute regulons of all TFs
    if gene_names is None:
        gene_names = np.unique(grn[tf_target_keys[0]].to_numpy()).tolist()

    regulon_dict = {}
    for gene in gene_names:

        gene_tf_bool = (grn[tf_target_keys[0]].to_numpy() == gene)

        if gene_tf_bool.sum() == 0:
            print(f'WARNING: Gene {gene} is not a TF in the given GRN')

        dummy = {'targets': grn[tf_target_keys[1]].to_numpy()[gene_tf_bool].tolist()}

        if additional_info_keys is not None:
            for key in additional_info_keys:
                dummy[key] = grn[key].to_numpy()[gene_tf_bool].tolist()

        regulon_dict[gene] = dummy

    return regulon_dict


def align_anndata_grn(
        adata: sc.AnnData,
        grn: pd.DataFrame,
        tf_target_keys: Tuple[str, str] = ('TF', 'target')
) -> Tuple[sc.AnnData, pd.DataFrame]:
    """
    Aligns the tabular scRNA-seq data with the input GRN such that only genes that are present in both remain.
    Args:
        adata (sc.AnnData): The input AnnData object containing gene expression data.
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        tf_target_keys (Tuple[str, str]): Column names for TFs and targets in the GRN. Defaults to ``('TF', 'target')``.
    Returns:
        Tuple[sc.AnnData, pd.DataFrame]: The aligned AnnData and DataFrame.
    """

    adata_genes = adata.var_names.to_numpy()
    grn_genes = np.unique(grn[list(tf_target_keys)].to_numpy())

    # Subset adata to genes that appear in GRN
    b = np.isin(adata_genes, grn_genes)
    adata = adata[:, b].copy()

    # Subset GRN to genes that appear in adata
    tf_bool = np.isin(grn[tf_target_keys[0]].to_numpy(), adata_genes)
    target_bool = np.isin(grn[tf_target_keys[1]].to_numpy(), adata_genes)
    grn_bool = tf_bool * target_bool
    grn = grn[grn_bool].copy()

    return adata, grn

