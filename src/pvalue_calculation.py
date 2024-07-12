
import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import *
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from .utils import labels_to_bool, solve_lsap


# Global p-values ######################################################################################################
def compute_westfall_young_adjusted_pvalues(adata: sc.AnnData,
                                            grn: pd.DataFrame,
                                            n_permutations: int = 100,
                                            weight_key: str = 'weight',
                                            cell_bool_key: str = 'cell_bool',
                                            clustering_dt_reg_key: str = 'cluster_bool_dt',
                                            clustering_obs_key: str = 'clusters') -> pd.DataFrame:
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
        # Todo: for global permutation, write this in inner loop:
        #  edge_labels = np.random.permutation(labels)[cell_bool]

        # Get clustering derived from the regressions stump during the weight calculation
        clustering_dt_reg = grn[clustering_dt_reg_key].iloc[i]

        for j in range(n_permutations):
            # Permute labels and compute weight
            permutation_weights[i, j] = solve_lsap(clust1=clustering_dt_reg,
                                                   clust2=np.random.permutation(edge_labels))
    # Compute empirical adjusted p-values
    true_weights = grn[weight_key].to_numpy()
    p_vals = test_statistic_to_w_y_adjusted_pvalue(true_weights=true_weights,
                                                   permutation_weights=permutation_weights)

    grn['pvals_wy'] = p_vals

    return grn


def compute_empirical_pvalues(adata: sc.AnnData,
                              grn: pd.DataFrame,
                              n_permutations: int = 100,
                              weight_key: str = 'weight',
                              cell_bool_key: str = 'cell_bool',
                              clustering_dt_reg_key: str = 'cluster_bool_dt',
                              clustering_obs_key: str = 'clusters') -> pd.DataFrame:
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
            permutation_weights[i, j] = solve_lsap(clust1=clustering_dt_reg,
                                                   clust2=np.random.permutation(edge_labels))

    # Compute empirical adjusted p-values
    true_weights = grn[weight_key].to_numpy()
    p_vals = test_statistic_to_emp_pvals(true_weights=true_weights,
                                         permutation_weights=permutation_weights,
                                         exact_pval=True)

    grn['emp_pvals'] = p_vals

    return grn


def adjust_pvals(grn: pd.DataFrame,
                 pval_key: str = 'pvals',
                 alpha: float = 0.05,
                 method: str = 'fdr_bh') -> pd.DataFrame:
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
    # - 'fdr_tsbh' : two stage fdr correction, independence or non-negative correlation
    # - 'fdr_tsbky' : two stage fdr correction, independence or non-negative correlation

    p_values = grn[pval_key].to_numpy()

    reject, pvals_corrected, _, _ = multipletests(pvals=p_values,
                                                  alpha=alpha,
                                                  method=method,
                                                  maxiter=1,
                                                  is_sorted=False,
                                                  returnsorted=False)
    grn[f'pvals_{method}'] = pvals_corrected

    return grn


def compute_corrected_pvalues(adata: sc.AnnData,
                              grn: pd.DataFrame,
                              method: str = 'wy',
                              n_permutations: int = 1000,
                              result_folder: Union[str, None] = None,
                              weight_key: str = 'weight',
                              cell_bool_key: str = 'cell_bool',
                              clustering_dt_reg_key: str = 'cluster_bool_dt',
                              clustering_obs_key: str = 'clusters',
                              plot: bool = False,
                              pval_key: Union[str, None] = None,
                              alpha: Union[float, None] = None,
                              fn_prefix: Union[str, None] = None) -> pd.DataFrame:

    # 'wy', 'bonferroni', 'sidak' control FWER, 'fdr_bh', 'fdr_by' control FDR
    assert method in {'wy', 'bonferroni', 'sidak', 'fdr_bh', 'fdr_by'}, \
        "Method can be 'wy', 'bonferroni', 'sidak', 'fdr_bh', 'fdr_by'"

    if method == 'wy':
        grn = compute_westfall_young_adjusted_pvalues(adata=adata,
                                                      grn=grn,
                                                      n_permutations=n_permutations,
                                                      weight_key=weight_key,
                                                      cell_bool_key=cell_bool_key,
                                                      clustering_dt_reg_key=clustering_dt_reg_key,
                                                      clustering_obs_key=clustering_obs_key)
    else:
        if pval_key is None:
            grn = compute_empirical_pvalues(adata=adata,
                                            grn=grn,
                                            n_permutations=n_permutations,
                                            weight_key=weight_key,
                                            cell_bool_key=cell_bool_key,
                                            clustering_dt_reg_key=clustering_dt_reg_key,
                                            clustering_obs_key=clustering_obs_key)
            pval_key = 'emp_pvals'

        if alpha is None:
            alpha = 0.05
        grn = adjust_pvals(grn=grn,
                           pval_key=pval_key,
                           alpha=alpha,
                           method=method)

    if result_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        grn_p = os.path.join(result_folder, f'{fn_prefix}grn.json')
        grn.to_json(grn_p)

    if plot:
        if alpha is None:
            alpha = 0.05
        weights = grn[weight_key].to_numpy()
        pvals = grn[f'pvals_{method}'].to_numpy()
        plt.scatter(weights, pvals, color='deepskyblue', marker='o', alpha=0.8)
        plt.xlabel('weight')
        plt.ylabel(f'{method} p-value')
        plt.axhline(y=alpha, color='red', label=f'alpha: {alpha}')
        plt.show()

    return grn


def remove_insignificant_edges(grn: pd.DataFrame,
                               alpha: float = 0.05,
                               p_value_key: str = 'pvals_wy',
                               result_folder: Union[None, str] = None,
                               verbosity: int = 0,
                               inplace: bool = True,
                               fn_prefix: Union[str, None] = None,
                               **kwargs) -> pd.DataFrame:

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
        print('### Removing edges due to insignificance ###')
        print(f'# There were {n_edges_before} edges in the GRN')
        print(f'# {grn.shape[0]} edges remain in the GRN, {n_edges_before - grn.shape[0]} edges were removed')

    return grn


# Auxiliary functions ##################################################################################################
def test_statistic_to_w_y_adjusted_pvalue(true_weights: np.ndarray,
                                          permutation_weights: np.ndarray) -> np.ndarray:
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


def test_statistic_to_emp_pvals(true_weights: np.ndarray,
                                permutation_weights: np.ndarray,
                                exact_pval: bool = True) -> np.ndarray:
    # Permutation_weights has dim: n_edges, n_permutations

    if exact_pval:
        # Fraction of times when permutation weight is bigger than true weight
        # Corrected by adding +1 in de-/nominator => No nonzero p-values (min_pval = 1 / n_permutations)
        # See paper: p-vals should never be zero ...
        p_vals = ((permutation_weights >= true_weights[:, np.newaxis]).sum(axis=1) + 1) / \
                 (permutation_weights.shape[1] + 1)
    else:
        # Fraction of times when permutation weight is bigger than true weight
        p_vals = (permutation_weights >= true_weights[:, np.newaxis]).sum(axis=1) / permutation_weights.shape[1]

    return p_vals
