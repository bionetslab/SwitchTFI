
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import *
from sklearn.tree import DecisionTreeRegressor

from .utils import csr_to_numpy, labels_to_bool, solve_lsap


def fit_regression_stump_model(
        adata: sc.AnnData,
        grn: pd.DataFrame,
        layer_key: Union[str, None] = None,
        result_folder: Union[str, None] = None,
        new_key: str = 'weight',
        clustering_obs_key: str = 'clusters',
        tf_target_keys: Tuple[str, str] = ('TF', 'target'),
        fn_prefix: Union[str, None] = None
) -> pd.DataFrame:
    """
    Fit a regression stump model to each TF-target edge in a GRN to compute the corresponding weight.

    This function fits a decision tree regressor of depth 1 (a regression stump) per edge to model the relationship between transcription factor and target gene. It calculates the decision threshold and the predicted values for each of the two resulting branches. Edge weights are computed based on the Jaccard similarity of the partitioning of cells induced by the decision boundary and existing progenitor-offspring annotations. The results are saved in the GRN DataFrame, including the fitted weight for each edge.

    Args:
        adata (sc.AnnData): The input AnnData object with scRNA-seq data.
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        layer_key (str, optional): The key for the expression data layer in AnnData. Defaults to None resulting in ``adata.X`` being used.
        result_folder (str, optional): Folder to save the resulting GRN. Defaults to None.
        new_key (str): The column name in GRN to store calculated weights. Defaults to ```'weight'```.
        clustering_obs_key (str): The key in `adata.obs` representing progenitor-offspring cluster labels. Defaults to ``'clusters'``.
        tf_target_keys (Tuple[str, str]): Column names in GRN representing TFs and targets. Defaults to ``('TF', 'target')``.
        fn_prefix (str, optional): Optional filename prefix when saving results. Defaults to None.

    Returns:
        pd.DataFrame: The updated GRN DataFrame with calculated weights for each TF-target pair.
    """

    tf_key = tf_target_keys[0]
    target_key = tf_target_keys[1]

    n_edges = grn.shape[0]
    cell_bools = [np.nan] * n_edges  # Store which cells were used for fitting the step function
    thresholds = [np.nan] * n_edges  # Store fitted threshold for each edge
    pred_l = [np.nan] * n_edges  # Store predicted value for inputs <= threshold (needed for plotting)
    pred_r = [np.nan] * n_edges  # Store predicted value for inputs >= threshold (needed for plotting)
    dt_reg_clusterings = [np.nan] * n_edges  # Store clustering derived by thresholding their tf-expression at threshold
    weights = [np.nan] * n_edges  # Store calculated weight for edges

    # Initialize decision tree regressor
    dt_regressor = DecisionTreeRegressor(
        criterion='squared_error',  # = variance (mean of values is prediction)
        splitter='best',  # unnecessary, have only one feature
        max_depth=1
    )

    for i in tqdm(range(n_edges), total=n_edges):
        # Get gene names of TF and target
        tf = grn[tf_key].iloc[i]
        target = grn[target_key].iloc[i]

        # Get expression vectors of TF and target
        try:
            if layer_key is None:
                x = csr_to_numpy(adata[:, tf].X).flatten()
                y = csr_to_numpy(adata[:, target].X).flatten()
            else:
                x = csr_to_numpy(adata[:, tf].layers[layer_key]).flatten()
                y = csr_to_numpy(adata[:, target].layers[layer_key]).flatten()
        except KeyError:
            weights[i] = 0
            warnings.warn(
                f'One of TF: {tf}, target: {target} appears in the GRN, but not in the Anndata object. '
                f'Setting {tf}-{target} edge weight to 0.',
                UserWarning
            )
            continue

        # Remove cells for which expression of TF or target is 0
        x, y, keep_bool = remove_zero_expression_cells(x=x, y=y)

        # Get label vector (C_1, C_2)
        labels = adata.obs[clustering_obs_key].to_numpy()[keep_bool]

        # Check for pathological cases
        pathological, same_label = check_for_pathological_cases(x=x, y=y, labels=labels)
        if pathological:
            # Set weight to min possible value == no explanatory power ...
            weights[i] = -1  # Lower than min possible weight of 0
            if same_label:
                weights[i] = 2  # Higher than max possible weight of 1
            continue

        # Reshape data to correct input format
        x = x.reshape((x.shape[0], 1))
        # Fit decision tree regressor of depth 1 (decision stump)
        dt_regressor.fit(X=x, y=y)
        # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

        # Update arrays with results
        cell_bools[i] = keep_bool
        if dt_regressor.tree_.value.flatten().shape[0] <= 1:  # Check for pathological cases that were not caught before
            weights[i] = -1
            continue
        else:
            thresholds[i] = dt_regressor.tree_.threshold[0]  # 0=root, 1=left, 2=right
            pred_l[i] = dt_regressor.tree_.value[1].flatten()[0]
            pred_r[i] = dt_regressor.tree_.value[2].flatten()[0]

        # Calculate predicted clusters L, R and resulting weight
        weights[i], dt_reg_clusterings[i] = calculate_weight(
            x_tf=x.flatten(),
            threshold=thresholds[i],
            labels=labels
        )

    grn['cell_bool'] = cell_bools
    grn['threshold'] = thresholds
    grn['pred_l'] = pred_l
    grn['pred_r'] = pred_r
    grn['cluster_bool_dt'] = dt_reg_clusterings
    grn[new_key] = weights

    # Sort grn dataframe w.r.t. 'weight'
    grn = grn.sort_values(by=[new_key], axis=0, ascending=False)
    grn = grn.reset_index(drop=True)

    if result_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        grn_p = os.path.join(result_folder, f'{fn_prefix}weighted_grn_all_edges.json')
        grn.to_json(grn_p)

    return grn


def prune_special_cases(
        grn: pd.DataFrame,
        result_folder: Union[str, None] = None,
        weight_key: str = 'weight',
        verbosity: int = 0,
        tf_target_keys: Tuple[str, str] = ('TF', 'target'),
        fn_prefix: Union[str, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prune edges from the GRN where no sensible weight could be fit.

    This function removes edges from the GRN that are considered pathological (e.g., if all cells have the same label and/or the regression stump could not be fit). The function keeps valid edges and updates the GRN by pruning invalid edges.

    Args:
        grn (pd.DataFrame): The GRN DataFrame with TF-target gene pairs.
        result_folder (str, optional): Folder to save the pruned GRN and special cases. Defaults to None.
        weight_key (str): Column name representing the edge weight. Defaults to ```'weight'```.
        verbosity (int): Level of logging. Defaults to 0.
        tf_target_keys (Tuple[str, str]): Column names for TF and target genes. Defaults to ``('TF', 'target')``.
        fn_prefix (str, optional): Optional filename prefix for saving the output. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The pruned GRN and the DataFrame of edges removed due to special cases.
    """

    if verbosity >= 1:
        n_edges_before = grn.shape[0]

    weights = grn[weight_key].to_numpy()

    same_label_bool = (weights == 2)
    same_label_pairs = grn[list(tf_target_keys)][same_label_bool]

    pathological_bool = (weights == -1)
    keep_bool = np.logical_not(np.logical_or(pathological_bool, same_label_bool))

    grn = grn[keep_bool]
    grn.reset_index(drop=True)

    if result_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        grn_p = os.path.join(result_folder, f'{fn_prefix}weighted_grn.json')
        grn.to_json(grn_p)
        samel_p = os.path.join(result_folder, f'{fn_prefix}same_label_edges.csv')
        same_label_pairs.to_csv(samel_p)

    if verbosity >= 1:
        print('### Removing edges that were special cases during weight fitting ###')
        print(f'# There were {n_edges_before} edges in the GRN')
        print(f'# {grn.shape[0]} edges remain in the GRN, {n_edges_before - grn.shape[0]} edges were removed')
        print(f'# {same_label_bool.sum()} of the removed were due to all cells having the same label')

    return grn, same_label_pairs


def prune_wrt_n_cells(
        grn: pd.DataFrame,
        mode: Literal['percent', 'quantile'] = 'percent',  # 'quantile'
        threshold: float = 0.05,
        result_folder: Union[str, None] = None,
        cell_bool_key: str = 'cell_bool',
        verbosity: int = 0,
        plot: bool = False,
        fn_prefix: Union[str, None] = None
) -> pd.DataFrame:
    """
    Prune edges from the GRN where too few cells were used during weight fitting.

    This function removes GRN edges for which too few cells were used to fit the regression stump model. The pruning is based on either a percentage of the maximum number of cells or a quantile of the distribution of cells used for fitting the decision stump model for an edge.

    Args:
        grn (pd.DataFrame): The GRN DataFrame with TF-target gene pairs.
        mode (str): The mode of pruning. Must be one of ``'percent'``, ``'quantile'``. Defaults to ``'percent'``.
        threshold (float): The threshold for pruning edges. Defaults to 0.05.
        result_folder (str, optional): Folder to save the pruned GRN. Defaults to None.
        cell_bool_key (str): The column in GRN with bool arrays encoding which cells were used for fitting. Defaults to ``'cell_bool'``.
        verbosity (int): Level of logging. Defaults to 0.
        plot (bool): Whether to plot the distribution of cell counts used for fitting. Defaults to False.
        fn_prefix (str, optional): Optional filename prefix when saving results. Defaults to None.

    Returns:
        pd.DataFrame: The pruned GRN DataFrame with edges removed based on cell count criteria.
    """

    # Remove edges from GRN for which too few cells were used for fitting the weight
    # -> 'quantile': remove threshold-quantile of edges with fewest cells, e.g. thresh=0.05, remove up to 0.5 quantile
    # -> 'percent': remove edges with less than threshold percent of the possible max n-cells

    n_edges_before = grn.shape[0]

    # Get array of bools (indicate which cells were used during weight fitting for the respective edge)
    cell_bool_array = np.vstack(grn[cell_bool_key])
    # For each edge compute the number of cell used for fitting the weight
    n_cells = cell_bool_array.sum(axis=1)

    if mode == 'percent':
        n_cells_max = n_cells.max()
        perc_thresh = n_cells_max * threshold
        keep_bool = (n_cells > perc_thresh)

    elif mode == 'quantile':
        # Compute threshold-quantile of n_cells
        q = np.quantile(n_cells, q=threshold, method='lower')
        keep_bool = (n_cells > q)

    else:
        raise ValueError('Mode must be either "percent" or "quantile".')

    if plot:

        fig, ax = plt.subplots(dpi=300)

        # Plot n-cell distribution
        ax.hist(
            n_cells,
            bins=30,
            edgecolor='grey',
            label='n_cells used for computing edge weight'
        )

        if mode == 'percent':
            ax.axvline(x=perc_thresh, color='red', label=f'thresh = max(n_cells) * {threshold}')
        if mode == 'quantile':
            ax.axvline(x=threshold, color='red', label=f'{threshold}-quantile')
        plt.legend()

        if fn_prefix is None:
            fn_prefix = ''

        if result_folder is not None:
            fig.savefig(os.path.join(result_folder, f'{fn_prefix}cells_per_edge_for_model_fitting.png'))

    # Prune GRN
    grn = grn[keep_bool].reset_index(drop=True)

    if (n_edges_before - grn.shape[0]) / n_edges_before > 0.5:
        print(f'WARNING: more than 50% ({round((n_edges_before - grn.shape[0]) / n_edges_before, 3)})'
              f'of the edges were removed')

    if verbosity >= 1:
        print('### Removing edges due to too few cell with non-zero expression during weight fitting ###')
        print(f'# There were {n_edges_before} edges in the GRN')
        print(f'# {grn.shape[0]} edges remain in the GRN, {n_edges_before - grn.shape[0]} edges were removed')

    if result_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        grn_p = os.path.join(result_folder, f'{fn_prefix}ncellpruned{threshold}{mode}_weighted_grn.json')
        grn.to_json(grn_p)

    return grn


def calculate_weights(
        adata: sc.AnnData,
        grn: pd.DataFrame,
        layer_key: Union[str, None] = None,
        result_folder: Union[str, None] = None,
        new_key: str = 'weight',
        n_cell_pruning_params: Union[Tuple[str, float], None] = ('percent', 0.2),
        clustering_obs_key: str = 'clusters',
        tf_target_keys: Tuple[str, str] = ('TF', 'target'),
        verbosity: int = 0,
        plot: bool = False,
        fn_prefix: Union[str, None] = None
) -> pd.DataFrame:
    """
    Calculate weights for GRN edges and prune edges based on special cases and cell count.

    This function performs the weight fitting step of the SwitchTFI method. It fits a regression stump model to the GRN, calculates edge weights, and prunes pathological edges (no sensible regression stump could be fit) and edges for which too few cells were used during weight fitting.

    Args:
        adata (sc.AnnData): The input AnnData object with gene expression data.
        grn (pd.DataFrame): The GRN DataFrame with TF-target gene pairs.
        layer_key (str, optional): The key for the expression data layer in AnnData. Defaults to None resulting in ``adata.X`` being used.
        result_folder (str, optional): Folder to save the final weighted GRN. Defaults to None.
        new_key (str): The column name in GRN to store the calculated weights. Defaults to ``'weight'``.
        n_cell_pruning_params (Tuple[str, float], optional): Parameters for pruning based on cell count per edge. Defaults to ``('percent', 0.2)``.
        clustering_obs_key (str): The key in ``adata.obs`` representing cluster labels. Defaults to ``'clusters'``.
        tf_target_keys (Tuple[str, str]): Column names in GRN representing TFs and targets. Defaults to ``('TF', 'target')``.
        verbosity (int): Level of logging. Defaults to 0.
        plot (bool): Whether to plot results of pruning. Defaults to False.
        fn_prefix (str, optional): Optional filename prefix when saving results. Defaults to None.

    Returns:
        pd.DataFrame: The final GRN DataFrame with calculated weights and all edges removed for which no weight fitting was possible.
    """

    grn = fit_regression_stump_model(
        adata=adata,
        grn=grn,
        layer_key=layer_key,
        result_folder=result_folder,
        new_key=new_key,
        clustering_obs_key=clustering_obs_key,
        tf_target_keys=tf_target_keys,
        fn_prefix=fn_prefix
    )

    # Weights are in [0,1] \cup {-1} \cup {2},
    # with
    # - w=-1 <=> pathological case
    # - w=2 <=> all cells for which TF, target are non-zero have the same label,
    #           i.e. edge only relevant in either progenitor or offspring cluster, not relevant for transition
    grn, _ = prune_special_cases(
        grn=grn,
        result_folder=result_folder,
        weight_key=new_key,
        verbosity=verbosity,
        tf_target_keys=tf_target_keys,
        fn_prefix=fn_prefix
    )

    if n_cell_pruning_params is not None:

        grn = prune_wrt_n_cells(
            grn=grn,
            mode=n_cell_pruning_params[0],
            threshold=n_cell_pruning_params[1],
            result_folder=result_folder,
            cell_bool_key='cell_bool',
            verbosity=verbosity,
            plot=plot,
            fn_prefix=fn_prefix
        )

    return grn


# Auxiliary functions ##################################################################################################
def remove_zero_expression_cells(
        x: np.ndarray,
        y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove cells where either the TF or target gene expression is zero.

    Args:
        x (np.ndarray): The expression values of the TF.
        y (np.ndarray): The expression values of the target gene.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered expression values of TF and target, along with a boolean array indicating which cells were kept.
    """

    keep_bool = np.logical_and((x != 0), (y != 0))

    return x[keep_bool], y[keep_bool], keep_bool


def check_for_pathological_cases(
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray
) -> Tuple[bool, bool]:
    """
    Check for pathological cases in the TF-target gene relationship.

    This function checks for pathological cases where the regression stump model cannot be fit due to issues such as all cells having the same label or all TF/target expression values being identical.

    Args:
        x (np.ndarray): The expression values of the transcription factor.
        y (np.ndarray): The expression values of the target gene.
        labels (np.ndarray): The cluster labels for the cells.

    Returns:
        Tuple[bool, bool]: A tuple indicating whether a pathological case was detected and whether all cells have the same label.
    """

    pathological = False
    same_label = False
    # Check if any cells remain after removing cells for which expression of TF and target is 0
    if x.size == 0:
        pathological = True
    # Check if all entries of x or y values are the same -> no sensible reg-tree can be fit
    elif np.all(x.flatten() == x.flatten()[0]) or np.all(y.flatten() == y.flatten()[0]):
        pathological = True

    # Check if all cells have the same cluster-label (all C_1 or all C_2)
    elif np.unique(labels).shape[0] <= 1:
        pathological = True
        same_label = True

    return pathological, same_label


def calculate_weight(
        x_tf: np.ndarray,
        threshold: float,
        labels: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Calculate the weight for a TF-target gene pair based on clustering similarity.

    This function computes the weight of a TF-target gene pair by comparing the progenitor-offspring cluster assignments of cells to the partition derived from thresholding TF expression. The threshold is the decision boundary of the regression stump (see ``fit_regression_stump_model()``).

    Args:
        x_tf (np.ndarray): The expression values of the transcription factor.
        threshold (float): The threshold value from the regression stump model.
        labels (np.ndarray): The cluster labels for the cells.

    Returns:
        Tuple[float, np.ndarray]: The weight for the TF-target gene pair and the binary
        clustering assignment based on TF expression.
    """

    # For each clustering there are only 2 Labels: C_1, C_2; L, R
    # -> Transform clustering vectors to bool form
    clustering_dt_regression = (x_tf <= threshold)
    clustering_cell_stage = labels_to_bool(labels)

    # Solve (trivial, only 2 possible cases) linear sum assignement problem (LSAP)
    # => Similarity score for the 2 clusterings (clustering1 = dt_reg, clustering2 = cell_stage)
    weight = solve_lsap(
        clust1=clustering_dt_regression,
        clust2=clustering_cell_stage
    )

    return weight, clustering_cell_stage
