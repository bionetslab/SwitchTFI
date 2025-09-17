
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Import functions to test
from switchtfi import (
    plot_grn,
    plot_regulon,
)



def test_plot_grn_runs_without_error(simulate_data, tmp_path):

    _, grn = simulate_data

    genes = np.unique(grn[['TF', 'target']].to_numpy())
    gc_df = pd.DataFrame(
        {
            'gene': genes,
            'centrality': np.random.uniform(size=genes.shape[0]),
        }
    )

    plot_file1 = tmp_path / 'test_grn.pdf'
    plot_file2 = tmp_path / 'test_grn.png'

    # Call function
    plot_grn(
        grn=grn,
        gene_centrality_df=gc_df,
        plot_folder=str(tmp_path),
        fn_prefix='test_',
    )

    plt.close('all')

    # Check file exists
    assert plot_file1.exists() or plot_file2.exists()


def test_plot_grn_with_ax(simulate_data, tmp_path):

    _, grn = simulate_data

    genes = np.unique(grn[['TF', 'target']].to_numpy())
    gc_df = pd.DataFrame(
        {
            'gene': genes,
            'centrality': np.random.uniform(size=genes.shape[0]),
        }
    )

    fig, ax = plt.subplots()

    plot_grn(
        grn=grn,
        gene_centrality_df=gc_df,
        plot_folder=tmp_path,
        fn_prefix='test_',
        ax=ax
    )

    # check if sth was drawn
    assert len(ax.images) > 0 or len(ax.collections) > 0

    plt.close('all')


def test_plot_regulon_outgoing(simulate_data):

    _, grn = simulate_data

    fig, ax = plt.subplots()
    plot_regulon(
        grn=grn,
        tf='TF0',
        sort_by='weight',
        top_k=6,
        ax=ax
    )

    assert len(ax.collections) > 0

    plt.close('all')


def test_plot_regulon_incoming(simulate_data):

    _, grn = simulate_data

    fig, ax = plt.subplots()

    ax = plot_regulon(
        grn=grn,
        tf='TF0',
        out=False,
        sort_by='pvals_wy',
        top_k=6,
        ax=ax
    )

    assert len(ax.collections) > 0

    plt.close('all')
