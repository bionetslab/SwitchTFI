
import pytest
import numpy as np

from switchtfi.tf_ranking import calculate_centrality_nx, rank_tfs, grn_to_nx


def test_grn_to_nx_builds_graph(simulate_data):
    _, grn = simulate_data
    g = grn_to_nx(grn)

    num_nodes = np.unique(grn[['TF', 'target']].to_numpy()).shape[0]
    assert g.number_of_nodes() == num_nodes
    assert g.number_of_edges() == grn.shape[0]
    for tf, target in grn[['TF', 'target']].itertuples(index=False):
        assert g.has_edge(tf, target)


@pytest.mark.parametrize(
    'centrality_measure',
    ['pagerank', 'out_degree', 'betweenness', 'closeness', 'katz']
)
def test_calculate_centrality_nx(simulate_data, centrality_measure):
    _, grn = simulate_data

    centrality_df, g = calculate_centrality_nx(
        grn=grn,
        centrality_measure=centrality_measure,
        weight_key=None,
        undirected=False,
    )

    # centrality dataframe should have expected columns
    assert 'gene' in centrality_df.columns
    assert centrality_measure in centrality_df.columns

    # graph should contain those genes
    assert set(centrality_df['gene']) == set(g.nodes)


def test_rank_tfs_returns_subset_of_tfs(simulate_data):
    _, grn = simulate_data
    tf_list = grn['TF'].unique()

    ranked_df = rank_tfs(
        grn=grn,
        centrality_measure='pagerank',
        undirected=True,
    )

    # result should contain all TFs
    assert set(ranked_df['gene']) == set(tf_list)
