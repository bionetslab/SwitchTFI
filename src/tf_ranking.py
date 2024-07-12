
import numpy as np
import pandas as pd
import networkx as nx
import os

from typing import *


def calculate_centrality_nx(grn: pd.DataFrame,
                            centrality_measure: str = 'pagerank',
                            reverse: bool = True,
                            undirected: bool = False,
                            weight_key: Union[str, None] = 'weight',
                            tf_target_keys: Tuple[str, str] = ('TF', 'target'),
                            **kwargs) -> Tuple[pd.DataFrame, nx.DiGraph]:

    if weight_key not in grn.columns and weight_key == 'score':
        weights = grn['weight'].to_numpy()
        pvals = grn['pvals_wy'].to_numpy()
        pvals += np.finfo(np.float64).eps
        grn['score'] = -np.log10(pvals) * weights
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

    # Assign pagerank values as node attributes
    nx.set_node_attributes(g, vertex_centrality_dict, name='pagerank')

    # Store pagerank values in pandas dataframe
    gene_list = [None] * len(vertex_centrality_dict)
    pagerank_list = [None] * len(vertex_centrality_dict)
    for i, (gene, pr_val) in enumerate(vertex_centrality_dict.items()):
        gene_list[i] = gene
        pagerank_list[i] = pr_val

    gene_pr_df = pd.DataFrame()
    gene_pr_df['gene'] = gene_list
    gene_pr_df[centrality_measure] = pagerank_list

    # Sort dataframe
    gene_pr_df = gene_pr_df.sort_values([centrality_measure], axis=0, ascending=False)
    gene_pr_df.reset_index(drop=True, inplace=True)

    return gene_pr_df, g


def rank_tfs(grn: pd.DataFrame,
             centrality_measure: str = 'pagerank',
             reverse: bool = True,
             undirected: bool = False,
             weight_key: Union[str, None] = None,
             result_folder: Union[str, None] = None,
             tf_target_keys: Tuple[str, str] = ('TF', 'target'),
             fn_prefix: Union[str, None] = None,
             **kwargs) -> pd.DataFrame:

    assert centrality_measure in {'pagerank', 'out_degree', 'eigenvector', 'closeness', 'betweenness', 'voterank',
                                  'katz'}, \
        "The 'centrality_measure' can be: 'pagerank', 'out_degree', 'eigenvector', 'closeness', 'betweenness', " \
        "'voterank', 'katz'"

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
def grn_to_nx(grn: pd.DataFrame,
              edge_attributes: Union[str, Tuple[str], bool, None] = 'weight',  # If True all columns will be added
              tf_target_keys: Tuple[str, str] = ('TF', 'target')) -> nx.DiGraph:

    if isinstance(edge_attributes, str):
        edge_attributes = (edge_attributes, )

    network = nx.from_pandas_edgelist(df=grn,
                                      source=tf_target_keys[0],
                                      target=tf_target_keys[1],
                                      edge_attr=edge_attributes,
                                      create_using=nx.DiGraph())

    return network
