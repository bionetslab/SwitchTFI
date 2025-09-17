
import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import json

from scipy.sparse import csr_matrix

from switchtfi.utils import (
    csr_to_numpy,
    anndata_to_numpy,
    load_grn_json,
    labels_to_bool,
    solve_lsap,
    calc_ji,
    get_regulons,
    align_anndata_grn,
)


def test_csr_to_numpy_and_anndata_to_numpy(simulate_data):
    adata, _ = simulate_data

    # raw .X
    arr1 = anndata_to_numpy(adata)
    assert isinstance(arr1, np.ndarray)
    assert arr1.shape == adata.X.shape

    # store to layer and fetch
    adata.layers['dummy'] = adata.X.copy()
    arr2 = anndata_to_numpy(adata, layer_key='dummy')
    assert np.allclose(arr1, arr2)

    # check csr_to_numpy explicit
    dense = np.array([[1, 0], [0, 1]])
    csr = csr_matrix(dense)
    assert np.allclose(csr_to_numpy(csr), dense)


def test_load_grn_json_roundtrip(tmp_path):
    df = pd.DataFrame({
        'TF': ['A', 'B'],
        'target': ['C', 'D'],
        'weights': [0.5, 0.8],
        'pvals': [0.01, 0.05],
        'list_col': [[1, 2], None],
    })
    f = tmp_path / 'grn.json'
    df.to_json(f)

    grn_loaded = load_grn_json(str(f))
    assert isinstance(grn_loaded, pd.DataFrame)
    assert set(grn_loaded.columns) == set(df.columns)
    assert isinstance(grn_loaded['list_col'].iloc[0], np.ndarray)


def test_labels_to_bool():
    labels1 = np.array(['A', 'B', 'A', 'B'])
    bools = labels_to_bool(labels1)
    assert bools.dtype == bool
    assert bools.sum() == 2

    labels2 = np.array(['A', 'B', 'C'])
    with pytest.raises(ValueError, match='at most 2 clusters.'):
        labels_to_bool(labels2)


def test_calc_ji():

    c1a = np.array([0, 0]).astype(bool)
    c1b = np.array([0, 0]).astype(bool)

    c2a = np.array([1, 0]).astype(bool)
    c2b = np.array([0, 1]).astype(bool)

    c3a = np.array([1, 0]).astype(bool)
    c3b = np.array([1, 0]).astype(bool)

    c4a = np.array([1, 1, 1]).astype(bool)
    c4b = np.array([1, 1, 0]).astype(bool)

    # empty sets => JI = 0
    assert calc_ji(c1a, c1b) == 0

    # disjoint sets => JI = 0
    assert calc_ji(c2a, c2b) == 0.0

    # completely overlapping sets => JI = 1.0
    assert calc_ji(c3a, c3b) == 1.0

    # partially overlapping sets => JI = 2 / 3
    assert calc_ji(c4a, c4b) == 2 / 3


def test_solve_lsap():

    a1 = np.array([1, 0]).astype(bool)
    b1 = np.array([1, 0]).astype(bool)
    s1 = solve_lsap(a1, b1)

    a2 = np.array([1, 0]).astype(bool)
    b2 = np.array([0, 1]).astype(bool)
    s2 = solve_lsap(a2, b2)

    a3 = np.array([1, 1]).astype(bool)
    b3 = np.array([0, 1]).astype(bool)
    s3 = solve_lsap(a3, b3)

    assert s1 == 1.0
    assert s2 == 1.0
    assert s3 == 0.25


def test_get_regulons(simulate_data):
    _, grn = simulate_data
    tfs = grn['TF'].unique().tolist()

    regs = get_regulons(grn, gene_names=tfs[:6], additional_info_keys=['weight', 'cell_bool'])

    assert isinstance(regs, dict)
    for tf in tfs[:6]:
        assert tf in regs
        assert 'targets' in regs[tf]
        assert isinstance(regs[tf]['targets'], list)
        assert all(isinstance(t, str) for t in regs[tf]['targets'])
        assert isinstance(regs[tf]['cell_bool'], list)
        assert all(isinstance(cb, np.ndarray) for cb in regs[tf]['cell_bool'])

    tf = tfs[0]
    targets = set(grn.loc[grn['TF'] == tf, 'target'])
    assert set(regs[tf]['targets']) == targets


def test_align_anndata_grn(simulate_data):
    adata, grn = simulate_data
    adata_aligned, grn_aligned = align_anndata_grn(adata, grn)

    # all genes in grn_aligned must be in adata_aligned.var_names
    adata_genes = set(adata_aligned.var_names)
    grn_genes = set(grn_aligned['TF']).union(set(grn_aligned['target']))
    assert grn_genes == adata_genes

    # shapes reduced compared to input
    assert adata_aligned.shape[1] <= adata.shape[1]
    assert grn_aligned.shape[0] <= grn.shape[0]
