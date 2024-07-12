
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import *
from sklearn.tree import DecisionTreeRegressor

from .utils import csr_to_numpy, labels_to_bool, solve_lsap


def fit_regression_stump_model(adata: sc.AnnData,
                               grn: pd.DataFrame,
                               layer_key: Union[str, None] = None,
                               result_folder: Union[str, None] = None,
                               new_key: str = 'weight',
                               clustering_obs_key: str = 'clusters',
                               tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                               fn_prefix: Union[str, None] = None) -> pd.DataFrame:
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
    dt_regressor = DecisionTreeRegressor(criterion='squared_error',  # = variance (mean of values is prediction)
                                         splitter='best',  # unnecessary, have only one feature
                                         max_depth=1)

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
            print(f'WARNING: one of TF: {tf}, target: {target} appears in the GRN, but not in the Anndata object')
            continue

        # Remove cells for which expression of TF and target is 0
        x, y, keep_bool = remove_double_zero_cells(x=x, y=y)

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
        weights[i], dt_reg_clusterings[i] = calculate_weight(x_tf=x.flatten(),
                                                             threshold=thresholds[i],
                                                             labels=labels)

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


def prune_special_cases(grn: pd.DataFrame,
                        result_folder: Union[str, None] = None,
                        weight_key: str = 'weight',
                        verbosity: int = 0,
                        tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                        fn_prefix: Union[str, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

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


def prune_wrt_n_cells(grn: pd.DataFrame,
                      mode: str = 'percent',  # 'quantile'
                      threshold: float = 0.05,
                      result_folder: Union[str, None] = None,
                      cell_bool_key: str = 'cell_bool',
                      verbosity: int = 0,
                      plot: bool = False,
                      fn_prefix: Union[str, None] = None) -> pd.DataFrame:

    # Remove edges from GRN for which too few cells were used for fitting the weight
    # -> 'quantile': remove threshold-quantile of edges with fewest cells
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
        keep_bool = np.zeros(n_edges_before).astype(bool)

    if plot:
        # Plot n-cell distribution
        plt.hist(n_cells, bins=30, label='n_cells used for computing edge weight')
        if mode == 'percent':
            plt.axvline(x=perc_thresh, color='red', label=f'thresh = max(n_cells) * {threshold}')
        if mode == 'quantile':
            plt.axvline(x=threshold, color='red', label=f'{threshold}-quantile')
        plt.legend()
        plt.show()

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


def calculate_weights(adata: sc.AnnData,
                      grn: pd.DataFrame,
                      layer_key: Union[str, None] = None,
                      result_folder: Union[str, None] = None,
                      new_key: str = 'weight',
                      n_cell_pruning_params: Union[Tuple[str, float], None] = ('percent', 0.2),
                      clustering_obs_key: str = 'clusters',
                      tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                      verbosity: int = 0,
                      plot: bool = False,
                      fn_prefix: Union[str, None] = None) -> pd.DataFrame:

    grn = fit_regression_stump_model(adata=adata,
                                     grn=grn,
                                     layer_key=layer_key,
                                     result_folder=result_folder,
                                     new_key=new_key,
                                     clustering_obs_key=clustering_obs_key,
                                     tf_target_keys=tf_target_keys,
                                     fn_prefix=fn_prefix)

    # Weights are in [0,1] \cup {-1} \cup {2},
    # with w=-1 <=> pathological case, w=2 <=> all cells for which TF, target are non-zero have the same label
    grn, _ = prune_special_cases(grn=grn,
                                 result_folder=result_folder,
                                 weight_key=new_key,
                                 verbosity=verbosity,
                                 tf_target_keys=tf_target_keys,
                                 fn_prefix=fn_prefix)

    if n_cell_pruning_params is not None:

        grn = prune_wrt_n_cells(grn=grn,
                                mode=n_cell_pruning_params[0],
                                threshold=n_cell_pruning_params[1],
                                result_folder=result_folder,
                                cell_bool_key='cell_bool',
                                verbosity=verbosity,
                                plot=plot,
                                fn_prefix=fn_prefix)
    return grn


# Auxiliary functions ##################################################################################################
def remove_double_zero_cells(x: np.ndarray,
                             y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    keep_bool = np.logical_and((x != 0), (y != 0))

    return x[keep_bool], y[keep_bool], keep_bool


def check_for_pathological_cases(x: np.ndarray,
                                 y: np.ndarray,
                                 labels: np.ndarray) -> Tuple[bool, bool]:
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


def calculate_weight(x_tf: np.ndarray,
                     threshold: float,
                     labels: np.ndarray) -> Tuple[float, np.ndarray]:
    # For each clustering there are only 2 Labels: C_1, C_2; L, R
    # -> Transform clustering vectors to bool form
    clustering_dt_regression = (x_tf <= threshold)
    clustering_cell_stage = labels_to_bool(labels)

    # Solve (trivial, only 2 possible cases) LSAP problem => Similarity score for the 2 clusterings
    # clust1 = dt_reg, clust2 = cell_stage
    weight = solve_lsap(clust1=clustering_dt_regression,
                        clust2=clustering_cell_stage)

    return weight, clustering_cell_stage
