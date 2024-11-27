
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from typing import *
from .weight_fitting import calculate_weights
from .pvalue_calculation import compute_corrected_pvalues, remove_insignificant_edges
from .tf_ranking import rank_tfs
from .plotting import plot_grn


def fit_model(adata: sc.AnnData,
              grn: pd.DataFrame,
              layer_key: Union[str, None] = 'magic_imputed',
              result_folder: Union[str, None] = None,
              weight_key: str = 'weight',
              n_cell_pruning_params: Union[Tuple[str, float], None] = ('percent', 0.2),
              pvalue_calc_method: str = 'wy',
              n_permutations: int = 1000,
              fwer_alpha: float = 0.05,
              centrality_measure: str = 'pagerank',
              reverse: bool = True,
              undirected: bool = False,
              centrality_weight_key: Union[str, None] = None,
              clustering_obs_key: str = 'clusters',
              tf_target_keys: Tuple[str, str] = ('TF', 'target'),
              verbosity: int = 0,
              plot: bool = False,
              save_intermediate: bool = False,
              fn_prefix: Union[str, None] = None,
              **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a gene regulatory network (GRN) model and rank transcription factors (TFs) based on centrality measures.

    This function aligns the input AnnData object with the GRN, calculates weights for the edges between
    TFs and target genes, computes corrected p-values for the weights using the specified permutation
    method, prunes insignificant edges, and ranks transcription factors based on centrality measures
    (e.g., PageRank, degree). The resulting transition GRN and ranked TFs can be saved and optionally plotted.

    Args:
        adata (sc.AnnData): The input AnnData object containing gene expression data.
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        layer_key (Union[str, None], optional): The key for the expression data layer to use. Defaults to 'magic_imputed'.
        result_folder (Union[str, None], optional): Folder to save the resulting GRN and ranked TFs. Defaults to None.
        weight_key (str, optional): Column name to store the calculated weights in the GRN. Defaults to 'weight'.
        n_cell_pruning_params (Union[Tuple[str, float], None], optional): Parameters for pruning of edges in the GRN
            based on the number of cells available for weight fitting. Defaults to ('percent', 0.2).
        pvalue_calc_method (str, optional): Method for p-value calculation ('wy', 'bonferroni', 'fdr_bh').
            Defaults to 'wy'.
        n_permutations (int, optional): Number of permutations for empirical p-value calculation. Defaults to 1000.
        fwer_alpha (float, optional): Significance threshold for FWER correction. Defaults to 0.05.
        centrality_measure (str, optional): Centrality measure to use for ranking TFs ('pagerank', 'out_degree', etc.). Defaults to 'pagerank'.
        reverse (bool, optional): Whether to reverse the direction of edges in the graph. Defaults to True.
        undirected (bool, optional): Whether to treat the graph as undirected. Defaults to False.
        centrality_weight_key (Union[str, None], optional): Column name for weights when calculating centrality. Defaults to None.
        clustering_obs_key (str, optional): Column name for clustering labels in `adata.obs`. Defaults to 'clusters'.
        tf_target_keys (Tuple[str, str], optional): Column names for TFs and targets in the GRN. Defaults to ('TF', 'target').
        verbosity (int, optional): Level of logging for detailed output. Defaults to 0.
        plot (bool, optional): Whether to plot the resulting GRN and centrality rankings. Defaults to False.
        save_intermediate (bool, optional): Whether to save intermediate results during the process. Defaults to False.
        fn_prefix (Union[str, None], optional): Optional prefix for filenames when saving results. Defaults to None.
        **kwargs: Additional arguments for the centrality calculation are passed to the respective NetworkX function.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The pruned GRN with significant edges and the ranked TFs based on centrality measures.
    """

    if fn_prefix is None:
        fn_prefix = ''

    if save_intermediate and result_folder is not None:
        interm_folder = result_folder
    elif save_intermediate:
        interm_folder = './'
    else:
        interm_folder = None

    adata, grn = align_anndata_grn(adata=adata,
                                   grn=grn,
                                   tf_target_keys=tf_target_keys)

    grn = calculate_weights(adata=adata,
                            grn=grn,
                            layer_key=layer_key,
                            result_folder=None,
                            new_key=weight_key,
                            n_cell_pruning_params=n_cell_pruning_params,
                            clustering_obs_key=clustering_obs_key,
                            tf_target_keys=tf_target_keys,
                            verbosity=verbosity,
                            plot=plot)

    grn = compute_corrected_pvalues(adata=adata,
                                    grn=grn,
                                    method=pvalue_calc_method,
                                    n_permutations=n_permutations,
                                    result_folder=interm_folder,
                                    weight_key=weight_key,
                                    cell_bool_key='cell_bool',
                                    clustering_dt_reg_key='cluster_bool_dt',
                                    clustering_obs_key=clustering_obs_key,
                                    plot=plot,
                                    alpha=fwer_alpha,
                                    fn_prefix=fn_prefix)

    grn = remove_insignificant_edges(grn=grn,
                                     alpha=fwer_alpha,
                                     p_value_key=f'pvals_{pvalue_calc_method}',
                                     result_folder=interm_folder,
                                     verbosity=verbosity,
                                     # weight_key=weight_key,
                                     fn_prefix=fn_prefix)

    ranked_tfs = rank_tfs(grn=grn,
                          centrality_measure=centrality_measure,
                          reverse=reverse,
                          undirected=undirected,
                          weight_key=centrality_weight_key,
                          result_folder=result_folder,
                          tf_target_keys=tf_target_keys,
                          fn_prefix=fn_prefix,
                          **kwargs)

    grn = grn[[tf_target_keys[0], tf_target_keys[1], weight_key, f'pvals_{pvalue_calc_method}']]

    if result_folder is not None:
        grn_p = os.path.join(result_folder, f'{fn_prefix}grn.csv')
        grn.to_csv(grn_p)

    if result_folder is not None:
        ax = None
        if plot:
            fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
        plot_grn(grn=grn,
                 gene_centrality_df=ranked_tfs.copy(),
                 plot_folder=result_folder,
                 weight_key=weight_key,
                 pval_key=f'pvals_{pvalue_calc_method}',
                 tf_target_keys=tf_target_keys,
                 axs=ax,
                 fn_prefix=fn_prefix)
        if plot:
            plt.show()

    return grn, ranked_tfs


# Auxiliary ############################################################################################################
def align_anndata_grn(adata: sc.AnnData,
                      grn: pd.DataFrame,
                      tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> Tuple[sc.AnnData, pd.DataFrame]:
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

