
import numpy as np
import pandas as pd
import scanpy as sc

from switchtfi.data_processing import (
    process_data,
    qc_cells_fct,
    qc_genes_fct,
    magic_imputation_fct,
    is_outlier,
)

def test_qc_cells(simulate_data):
    # cell QC
    adata, _ = simulate_data
    adata_cells = qc_cells_fct(adata, verbosity=0)
    assert isinstance(adata_cells, sc.AnnData)
    assert adata_cells.n_obs <= adata.n_obs


def test_qc_genes(simulate_data):
    # gene QC
    adata, _ = simulate_data
    adata_genes = qc_genes_fct(adata, verbosity=0)
    assert isinstance(adata_genes, sc.AnnData)
    assert adata_genes.n_vars <= adata.n_vars


def test_is_outlier(simulate_data):
    # Add a fake QC metric
    adata, _ = simulate_data
    adata.obs['fake_qc'] = np.concatenate([np.array([1000]), np.zeros(adata.n_obs - 1)])
    outliers = is_outlier(adata, 'fake_qc', nmads=3)
    assert outliers.sum() == 1
    assert isinstance(outliers, pd.Series)


def test_magic_imputation(simulate_data):

    # Add normalized + log1p layer manually
    adata, _ = simulate_data
    adata.layers['X_log1p_norm'] = adata.X.copy()

    num_zero = (adata.layers['X_log1p_norm'] == 0).sum()

    out = magic_imputation_fct(adata, verbosity=0)

    assert 'X_magic_imputed' in out.layers
    assert out.layers['X_magic_imputed'].shape == out.shape
    assert (out.layers['X_magic_imputed'] == 0).sum() <= num_zero


def test_process_data_pipeline(simulate_data):

    adata, _ = simulate_data
    adata.layers['X_log1p_norm'] = adata.X.copy()

    processed = process_data(adata, magic_imputation=False, verbosity=0)

    # Check layers created
    assert 'X_normalized' in processed.layers
    assert 'X_log1p_norm' in processed.layers
    assert 'X_log1p_norm' in processed.layers



