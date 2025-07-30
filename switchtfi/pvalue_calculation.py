
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

from typing import *
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from .utils import labels_to_bool, solve_lsap


# Global p-values ######################################################################################################
def compute_westfall_young_adjusted_pvalues(
        adata: sc.AnnData,
        grn: pd.DataFrame,
        n_permutations: int = 100,
        weight_key: str = 'weight',
        cell_bool_key: str = 'cell_bool',
        clustering_dt_reg_key: str = 'cluster_bool_dt',
        clustering_obs_key: str = 'clusters'
) -> pd.DataFrame:
    """
    Compute Westfall-Young adjusted p-values for the edges weights fitted to a GRN by SwitchTFI.

    This function performs label permutation to compute empirical p-values for each edge in the GRN
    using the Westfall-Young method. It fits a regression model and in term a weight for each permutation
    and calculates the maximum test statistic over all edges of the GRN. The empirical p-value is computed
    by counting how often the maximum test statistic is more extrem than the true weight. The resulting p-values
    are adjusted, they control the family-wise error rate (FWER).

    Args:
        adata (sc.AnnData): The input AnnData object.
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        n_permutations (int): Number of permutations for the Westfall-Young procedure. Defaults to 100.
        weight_key (str): Column name in the GRN representing the weights. Defaults to 'weight'.
        cell_bool_key (str): Column name in the GRN containing bool arrays indicating which cells were used during
        weight fitting for the respective edge. Defaults to 'cell_bool'.
        clustering_dt_reg_key (str): Column name in the GRN, containing the arrays with entries
        corresponding to the clustering derived during weight calculation. Defaults to 'cluster_bool_dt'.
        clustering_obs_key (str): Key for the cluster labels in `adata.obs`. Defaults to 'clusters'.

    Returns:
        pd.DataFrame: The GRN with adjusted p-values added in the 'pvals_wy' column.
    """

    # Iterate (for n_permutations):
    #   - Permute labels
    #   - Fit model => weights (for all edges)
    #   - For those weights store max(weight)
    # - For original weights count #[weight >= max(weight)] -> empirical p-value
    # See paper -> Limit FWER

    n_edges = grn.shape[0]
    # Get labels fom anndata, turn into bool vector
    labels = labels_to_bool(adata.obs[clustering_obs_key].to_numpy())
    # Initialize container to store the weights computed with permuted labels
    permutation_weights = np.zeros((n_edges, n_permutations))
    # Iterate over edges
    for i in tqdm(range(n_edges), total=n_edges):
        # Get labels of the cells that were used for fitting the weight
        cell_bool = grn[cell_bool_key].iloc[i]
        edge_labels = labels[cell_bool]

        # Note: for global permutation, write this in inner loop:
        #  edge_labels = np.random.permutation(labels)[cell_bool]

        # Get clustering derived from the regressions stump during the weight calculation
        clustering_dt_reg = grn[clustering_dt_reg_key].iloc[i]

        for j in range(n_permutations):
            # Permute labels and compute weight
            permutation_weights[i, j] = solve_lsap(
                clust1=clustering_dt_reg,
                clust2=np.random.permutation(edge_labels)
            )

    # Compute empirical adjusted p-values
    true_weights = grn[weight_key].to_numpy()
    p_vals = test_statistic_to_w_y_adjusted_pvalue(
        true_weights=true_weights,
        permutation_weights=permutation_weights
    )

    grn['pvals_wy'] = p_vals

    return grn


def compute_empirical_pvalues(
        adata: sc.AnnData,
        grn: pd.DataFrame,
        n_permutations: int = 100,
        weight_key: str = 'weight',
        cell_bool_key: str = 'cell_bool',
        clustering_dt_reg_key: str = 'cluster_bool_dt',
        clustering_obs_key: str = 'clusters'
) -> pd.DataFrame:

    n_edges = grn.shape[0]
    # Get labels fom anndata, turn into bool vector
    labels = labels_to_bool(adata.obs[clustering_obs_key].to_numpy())
    # Initialize container to store the weights computed with permuted labels
    permutation_weights = np.zeros((n_edges, n_permutations))
    # Iterate over edges
    for i in tqdm(range(n_edges), total=n_edges):
        # Get labels of the cells that were used for fitting the weight
        cell_bool = grn[cell_bool_key].iloc[i]
        edge_labels = labels[cell_bool]

        # Get clustering derived from the regressions stump during the weight calculation
        clustering_dt_reg = grn[clustering_dt_reg_key].iloc[i]

        for j in range(n_permutations):
            # Permute labels and compute weight
            permutation_weights[i, j] = solve_lsap(
                clust1=clustering_dt_reg,
                clust2=np.random.permutation(edge_labels)
            )

    # Compute empirical adjusted p-values
    true_weights = grn[weight_key].to_numpy()
    p_vals = test_statistic_to_emp_pvals(
        true_weights=true_weights,
        permutation_weights=permutation_weights,
        exact_pval=True
    )

    grn['emp_pvals'] = p_vals

    return grn


def adjust_pvals(
        grn: pd.DataFrame,
        pval_key: str = 'pvals',
        method: str = 'fdr_bh',
        alpha: Union[float, None] = None,
) -> pd.DataFrame:

    # For FWER control use
    # - 'bonferroni': very conservative, no assumptions ...
    # - 'sidak': one-step correction, independence (or others e.g. normality of test statistics of individual tests)
    # - 'holm-sidak': step down method using Sidak adjustments, same as for Sidak
    # - 'holm' : step-down method using Bonferroni adjustments, no assumptions
    # - 'simes-hochberg' : step-up method, independence
    # - 'hommel' : closed method based on Simes tests, non-negative correlation
    # For FDR control use:
    # - 'fdr_bh' : Benjamini/Hochberg, independence or non-negative correlation
    # - 'fdr_by' : Benjamini/Yekutieli, independence or negative correlation
    # - 'fdr_tsbh' : two stage fdr correction, independence or non-negative correlation, uses alpha
    # - 'fdr_tsbky' : two stage fdr correction, independence or non-negative correlation, uses alpha

    # NOTE: alpha is only used for two-stage procedures!

    if alpha is None and method in {'fdr_tsbh', 'fdr_tsbky'}:
        alpha = 0.05
        warnings.warn(
            f'The selected method {method} requires alpha to be set. A default of 0.05 is used.',
            UserWarning
        )

    p_values = grn[pval_key].to_numpy()

    reject, pvals_corrected, _, _ = multipletests(
        pvals=p_values,
        alpha=alpha,
        method=method,
        maxiter=1,
        is_sorted=False,
        returnsorted=False
    )

    grn[f'pvals_{method}'] = pvals_corrected

    return grn


def compute_corrected_pvalues(
        adata: sc.AnnData,
        grn: pd.DataFrame,
        method: Literal['wy', 'bonferroni', 'sidak', 'fdr_bh', 'fdr_by'] = 'wy',
        n_permutations: int = 1000,
        result_folder: Union[str, None] = None,
        weight_key: str = 'weight',
        cell_bool_key: str = 'cell_bool',
        clustering_dt_reg_key: str = 'cluster_bool_dt',
        clustering_obs_key: str = 'clusters',
        plot: bool = False,
        pval_key: Union[str, None] = None,
        fn_prefix: Union[str, None] = None
) -> pd.DataFrame:
    """
    Compute corrected p-values for GRN edges using simple permutation-based empirical p-values
    plus multiple testing correction or the Westfall-Young method.

    This function computes either Westfall-Young adjusted p-values or applies a multiple testing correction
    to empirical p-values based on the specified method. It updates the GRN with the corrected p-values
    and optionally saves the results.

    Args:
        adata (sc.AnnData): The input AnnData object.
        grn (pd.DataFrame): The GRN DataFrame with TF-target gene pairs.
        method (Literal['wy', 'bonferroni', 'sidak', 'fdr_bh', 'fdr_by'] ): P-value calculation method. Defaults to 'wy'.
        n_permutations (int): Number of permutations for Westfall-Young or empirical p-values. Defaults to 1000.
        result_folder (str, optional): Folder to save the results. Defaults to None.
        weight_key (str): Column name for the weights in the GRN. Defaults to 'weight'.
        cell_bool_key (str): Column name in the GRN containing a bool arrays indicating which cells
        were used for weight fitting. Defaults to 'cell_bool'.
        clustering_dt_reg_key (str): Column name in the GRN, containing the arrays with entries corresponding to the
        clustering derived during weight calculation. Defaults to 'cluster_bool_dt'.
        clustering_obs_key (str): Key for cluster labels in `adata.obs`. Defaults to 'clusters'.
        plot (bool): Whether to generate a scatter plot of weights vs. p-values. Defaults to False.
        pval_key (str, optional): Column name for empirical p-values (not multiple testing corrected) in the GRN,
        if it exists. Defaults to None, i.e. empirical p-values are computed from scratch.
        Defaults to None, i.e. is set to 0.05 internally.
        fn_prefix (str, optional): Optional filename prefix for saving results. Defaults to None.

    Returns:
        pd.DataFrame: The GRN with corrected p-values added.
    """

    # 'wy', 'bonferroni', 'sidak' control FWER, 'fdr_bh', 'fdr_by' control FDR
    if method not in {'wy', 'bonferroni', 'sidak', 'fdr_bh', 'fdr_by'}:
        raise ValueError("Method must be one of: 'wy', 'bonferroni', 'sidak', 'fdr_bh', 'fdr_by'")

    if method == 'wy':
        grn = compute_westfall_young_adjusted_pvalues(
            adata=adata,
            grn=grn,
            n_permutations=n_permutations,
            weight_key=weight_key,
            cell_bool_key=cell_bool_key,
            clustering_dt_reg_key=clustering_dt_reg_key,
            clustering_obs_key=clustering_obs_key
        )

    else:
        if pval_key is None:
            grn = compute_empirical_pvalues(
                adata=adata,
                grn=grn,
                n_permutations=n_permutations,
                weight_key=weight_key,
                cell_bool_key=cell_bool_key,
                clustering_dt_reg_key=clustering_dt_reg_key,
                clustering_obs_key=clustering_obs_key
            )

            pval_key = 'emp_pvals'

        grn = adjust_pvals(
            grn=grn,
            pval_key=pval_key,
            method=method,
            alpha=None,
        )

    if result_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        grn_p = os.path.join(result_folder, f'{fn_prefix}grn.json')
        grn.to_json(grn_p)

    if plot and result_folder is not None:

        weights = grn[weight_key].to_numpy()
        pvals = grn[f'pvals_{method}'].to_numpy()
        fig, ax = plt.subplots(dpi=300)
        ax.scatter(weights, pvals, color='deepskyblue', marker='o', alpha=0.8)
        ax.set_xlabel('weight')
        ax.set_ylabel(f'{method} corrected p-value')
        ax.axhline(y=0.05, color='red', label='alpha=0.05')
        ax.axhline(y=0.01, color='orange', label='alpha=0.01')
        plt.legend()
        fig.savefig(os.path.join(result_folder, f'weight_vs_{method}_corrected_pvalues.png'))
        plt.close(fig)

    else:
        warnings.warn('Saving of plot not possible if result_folder is not specified.', UserWarning)

    return grn


def remove_insignificant_edges(
        grn: pd.DataFrame,
        alpha: float = 0.05,
        p_value_key: str = 'pvals_wy',
        result_folder: Union[str, None] = None,
        verbosity: int = 0,
        inplace: bool = True,
        fn_prefix: Union[str, None] = None
) -> pd.DataFrame:
    """
    Remove edges with insignificant weight from the GRN based on previously computed adjusted p-values.

    This function removes edges from the GRN where the adjusted p-value exceeds the
    specified significance level (alpha). The remaining significant edges are returned, and optionally,
    the results are saved to a file.

    Args:
        grn (pd.DataFrame): The GRN DataFrame containing edges and their adjusted p-values.
        alpha (float): The significance threshold for removing edges. Defaults to 0.05.
        p_value_key (str): Column name for the p-values to evaluate. Defaults to 'pvals_wy'.
        result_folder (str, optional): Folder to save the filtered GRN. Defaults to None.
        verbosity (int): Level of logging for detailed output. Defaults to 0.
        inplace (bool): Whether to modify the GRN in place or return a copy. Defaults to True.
        fn_prefix (str, optional): Optional filename prefix for saving results. Defaults to None.

    Returns:
        pd.DataFrame: The filtered GRN containing only significant edges.
    """

    if not inplace:
        grn = grn.copy(deep=True)

    n_edges_before = grn.shape[0]
    keep_bool = grn[p_value_key].to_numpy() <= alpha
    grn = grn[keep_bool]

    grn = grn.sort_values(by=[p_value_key], axis=0, ascending=True)
    grn = grn.reset_index(drop=True)

    if result_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        grn_p = os.path.join(result_folder, f'{fn_prefix}sig{p_value_key}{alpha}_grn.json')
        grn.to_json(grn_p)

    if verbosity >= 1:
        print('### Removing edges below FWER threshold ###')
        print(
            f'# Out of {n_edges_before} edges {grn.shape[0]} edges remain in the GRN,\n'
            f'{n_edges_before - grn.shape[0]} edges were removed'
        )

    return grn


# Auxiliary functions ##################################################################################################
def test_statistic_to_w_y_adjusted_pvalue(
        true_weights: np.ndarray,
        permutation_weights: np.ndarray
) -> np.ndarray:

    # Get maximum weight (test statistic) per permutation, dim: n_edges, n_permutations
    max_permutation_weight = permutation_weights.max(axis=0)

    p_vals = (true_weights[:, np.newaxis] <= max_permutation_weight).sum(axis=1) / permutation_weights.shape[1]

    return p_vals


def test_statistic_to_w_y_adjusted_pvalue2(true_weights: np.ndarray,
                                           permutation_weights: np.ndarray) -> np.ndarray:
    # Get maximum weight (test statistic) per permutation, dim: n_edges, n_permutations
    max_permutation_weight = permutation_weights.max(axis=0)

    p_vals = ((true_weights[:, np.newaxis] <= max_permutation_weight).sum(axis=1) + 1) / \
             (permutation_weights.shape[1] + 1)

    return p_vals


def test_statistic_to_emp_pvals(
        true_weights: np.ndarray,
        permutation_weights: np.ndarray,
        exact_pval: bool = True
) -> np.ndarray:

    # Permutation_weights has dim: n_edges, n_permutations

    if exact_pval:
        # Fraction of times when permutation weight is bigger than true weight
        # Corrected by adding +1 in de-/nominator => No nonzero p-values (min_pval = 1 / n_permutations)
        # See paper: p-vals should never be zero ...
        p_vals = (
                ((permutation_weights >= true_weights[:, np.newaxis]).sum(axis=1) + 1) /
                (permutation_weights.shape[1] + 1)
        )
    else:
        # Fraction of times when permutation weight is bigger than true weight
        p_vals = (permutation_weights >= true_weights[:, np.newaxis]).sum(axis=1) / permutation_weights.shape[1]

    return p_vals
