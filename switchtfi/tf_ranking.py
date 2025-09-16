
import os
import numpy as np
import pandas as pd
import networkx as nx

from typing import *


def calculate_centrality_nx(
        grn: pd.DataFrame,
        centrality_measure: str = 'pagerank',
        reverse: bool = True,
        undirected: bool = False,
        weight_key: Union[str, None] = 'weight',
        tf_target_keys: Tuple[str, str] = ('TF', 'target'),
        **kwargs
) -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    Calculate centrality values for nodes in the directed or undirected, weighted or unweighted GRN using NetworkX.

    Args:
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        centrality_measure (str): The centrality measure to compute. Must be one of ``'pagerank'``, ``'out_degree'``, ``'betweenness'``, ``'closeness'``, ``'eigenvector'``, ``'katz'``, or ``'voterank'``. Defaults to ``'pagerank'``.
        reverse (bool): Whether to reverse the direction of edges in the graph. Defaults to True.
        undirected (bool): Whether to treat the graph as undirected. Defaults to False.
        weight_key (str, optional): The key for edge weights in the GRN. None corresponds to the unweighted case. Defaults to ``'weight'``.
        tf_target_keys (Tuple[str, str]): Column names for TF and target genes. Defaults to ``('TF', 'target')``.
        **kwargs: Additional arguments for the centrality calculation are passed to the respective NetworkX function.

    Returns:
        Tuple[pd.DataFrame, nx.DiGraph]: A DataFrame with genes and their centrality scores, and the NetworkX graph.
    """

    # Calculate the score
    if weight_key not in grn.columns and weight_key == 'score':
        weights = grn['weight'].to_numpy()
        pvals = grn['pvals_wy'].to_numpy()
        pvals += np.finfo(np.float64).eps
        grn['score'] = -np.log10(pvals) * weights

    # calculate the -log-transformed p-values
    elif weight_key not in grn.columns and weight_key == '-log_pvals':
        pvals = grn['pvals_wy'].to_numpy()
        pvals += np.finfo(np.float64).eps
        grn['-log_pvals'] = - np.log10(pvals)

    g = grn_to_nx(grn=grn, edge_attributes=weight_key, tf_target_keys=tf_target_keys)

    if reverse:
        g = g.reverse(copy=True)

    if undirected:
        g = g.to_undirected()

    if centrality_measure == 'pagerank':
        vertex_centrality_dict = nx.pagerank(g, weight=weight_key, **kwargs)
    elif centrality_measure == 'out_degree':
        vertex_centrality_dict = dict(g.out_degree(weight=weight_key))
    elif centrality_measure == 'eigenvector':
        vertex_centrality_dict = nx.eigenvector_centrality(g, weight=weight_key, **kwargs)
    elif centrality_measure == 'closeness':
        vertex_centrality_dict = nx.closeness_centrality(g, distance=weight_key, **kwargs)
    elif centrality_measure == 'betweenness':
        vertex_centrality_dict = nx.betweenness_centrality(g, weight=weight_key, **kwargs)
    elif centrality_measure == 'voterank':
        dummy = nx.voterank(g)
        vertex_centrality_dict = {}
        for i, v in enumerate(dummy):
            vertex_centrality_dict[v] = len(dummy) - i
    elif centrality_measure == 'katz':
        vertex_centrality_dict = nx.katz_centrality(g, weight=weight_key, **kwargs)
    else:
        vertex_centrality_dict = {}

    # Assign centrality values as node attributes
    nx.set_node_attributes(g, vertex_centrality_dict, name=f'{centrality_measure}_centrality')

    # Store centrality values in pandas dataframe
    gene_list = [None] * len(vertex_centrality_dict)
    centrality_list = [None] * len(vertex_centrality_dict)
    for i, (gene, centrality_value) in enumerate(vertex_centrality_dict.items()):
        gene_list[i] = gene
        centrality_list[i] = centrality_value

    gene_pr_df = pd.DataFrame()
    gene_pr_df['gene'] = gene_list
    gene_pr_df[centrality_measure] = centrality_list

    # Sort dataframe
    gene_pr_df = gene_pr_df.sort_values([centrality_measure], axis=0, ascending=False)
    gene_pr_df.reset_index(drop=True, inplace=True)

    return gene_pr_df, g


def rank_tfs(
        grn: pd.DataFrame,
        centrality_measure: Literal[
            'pagerank', 'out_degree', 'eigenvector', 'closeness', 'betweenness', 'voterank', 'katz'
        ] = 'pagerank',
        reverse: bool = True,
        undirected: bool = False,
        weight_key: Union[str, None] = None,
        result_folder: Union[str, None] = None,
        tf_target_keys: Tuple[str, str] = ('TF', 'target'),
        fn_prefix: Union[str, None] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Rank transcription factors (TFs) in the GRN based on their centrality in the network.

    This function ranks transcription factors in the GRN based on a specified centrality measure
    (e.g., PageRank, degree, closeness). The function computes the centrality measure for all genes
    and filters the result to retain only TFs. Optionally the all edges can be reversed, all edges
    can be set to be undirected and edge weights can be defined before computing centrality.

    Args:
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        centrality_measure (Literal['pagerank', 'out_degree', 'eigenvector', 'closeness', 'betweenness', 'voterank', 'katz']): The centrality measure to use for ranking. Defaults to 'pagerank'.
        reverse (bool): Whether to reverse the direction of edges in the graph. Defaults to True.
        undirected (bool): Whether to treat the graph as undirected. Defaults to False.
        weight_key (str, optional): The key for edge weights in the GRN. None corresponds to the unweighted case. If the key 'score' is passed and does not already exist in the GRN, then the weight is computed as the score = -log10(p-vals) * weight. Defaults to None.
        result_folder (str, optional): Folder to save the ranked TFs. Defaults to None.
        tf_target_keys (Tuple[str, str]): Column names for TF and target genes. Defaults to ('TF', 'target').
        fn_prefix (str, optional): Optional filename prefix for saving results. Defaults to None.
        **kwargs: Additional arguments for the centrality calculation are passed to the respective NetworkX function.

    Returns:
        pd.DataFrame: A DataFrame of ranked transcription factors based on the selected centrality measure.
    """

    if centrality_measure not in {
        'pagerank', 'out_degree', 'eigenvector', 'closeness', 'betweenness', 'voterank', 'katz'
    }:
        raise ValueError(
            "The 'centrality_measure' must be one of: "
            "'pagerank', 'out_degree', 'eigenvector', 'closeness', 'betweenness', 'voterank', 'katz'"
        )

    # Compute pagerank of all genes in GRN
    gene_pr_df, _ = calculate_centrality_nx(grn=grn,
                                            centrality_measure=centrality_measure,
                                            reverse=reverse,
                                            undirected=undirected,
                                            weight_key=weight_key,
                                            tf_target_keys=tf_target_keys,
                                            **kwargs)

    # Remove genes that are not TFs
    tfs = grn[tf_target_keys[0]].to_numpy()
    tf_bool = np.isin(gene_pr_df['gene'].to_numpy(), tfs)

    # print('### All genes', gene_pr_df.shape)
    gene_pr_df = gene_pr_df[tf_bool]
    gene_pr_df.reset_index(drop=True)
    # print('### Only TFs', gene_pr_df.shape)

    if result_folder is not None:
        if fn_prefix is None:
            fn_prefix = ''
        res_p = os.path.join(result_folder, f'{fn_prefix}ranked_tfs.csv')
        gene_pr_df.to_csv(res_p)

    return gene_pr_df


# Auxiliary ############################################################################################################
def grn_to_nx(
        grn: pd.DataFrame,
        edge_attributes: Union[str, Tuple[str], bool, None] = 'weight',  # If True all columns will be added
        tf_target_keys: Tuple[str, str] = ('TF', 'target')
) -> nx.DiGraph:
    """
    Convert a GRN DataFrame into a NetworkX graph.

    Args:
        grn (pd.DataFrame): The GRN DataFrame containing TF-target gene pairs.
        edge_attributes (Union[str, Tuple[str], bool], optional): Edge attributes (columns) to include in the graph. If True, all columns are added. Defaults to ```'weight'```.
        tf_target_keys (Tuple[str, str]): Column names for TF and target genes. Defaults to ``('TF', 'target')``.

    Returns:
        nx.DiGraph: A NetworkX directed graph representing the GRN.
    """

    if isinstance(edge_attributes, str):
        edge_attributes = (edge_attributes, )

    elif isinstance(edge_attributes, bool) and edge_attributes is True:
        edge_attributes = grn.columns.drop(list(tf_target_keys)).tolist()

    elif edge_attributes in {False, None}:
        edge_attributes = None

    network = nx.from_pandas_edgelist(
        df=grn,
        source=tf_target_keys[0],
        target=tf_target_keys[1],
        edge_attr=edge_attributes,
        create_using=nx.DiGraph,
    )

    return network
