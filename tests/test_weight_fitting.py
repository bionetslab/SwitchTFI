
import numpy as np
import pandas as pd
import pytest

from switchtfi.weight_fitting import (
    remove_zero_expression_cells,
    check_for_pathological_cases,
    calculate_weight,
    fit_regression_stump_model,
    prune_special_cases,
    prune_wrt_n_cells,
    calculate_weights,
)


# -------------------
# Auxiliary functions
# -------------------

def test_remove_double_zero_cells():
    x = np.array([0, 1, 0, 2])
    y = np.array([0, 3, 4, 0])
    x_out, y_out, keep = remove_zero_expression_cells(x, y)

    assert len(x_out) == len(y_out)
    assert np.all(x_out != 0) and np.all(y_out != 0)
    assert keep.sum() == 1


def test_check_for_pathological_cases_same_label():
    x1 = np.array([])
    y1 = np.array([])
    l1 = np.array([])
    p1, sl1 = check_for_pathological_cases(x1, y1, l1)

    x2 = np.array([1, 1, 1])
    y2 = np.array([1, 2, 3])
    l2 = np.array(['Prog', 'Prog', 'Off'])
    p2, sl2 = check_for_pathological_cases(x2, y2, l2)

    x3 = np.array([1, 3, 2])
    y3 = np.array([1, 2, 3])
    l3 = np.array(['Prog', 'Prog', 'Prog'])
    p3, sl3 = check_for_pathological_cases(x3, y3, l3)

    assert p1
    assert p2
    assert p3 and sl3


def test_calculate_weight():
    x_tf = np.array([0.1, 0.1, 0.1, 0.9])
    labels = np.array(['A', 'A', 'B', 'B'])
    weight, clust = calculate_weight(x_tf, threshold=0.5, labels=labels)

    assert 0.25 <= weight <= 1.0
    assert clust.shape == labels.shape
    assert clust.sum() == 2


def test_calculate_max_weight():
    x_tf = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array(['A', 'A', 'B', 'B'])
    weight, clust = calculate_weight(x_tf, threshold=0.5, labels=labels)

    assert weight == 1.0


# -------------------
# Integration tests with simulate_data fixture
# -------------------

def test_fit_regression_stump_model(simulate_data):
    adata, grn = simulate_data
    out = fit_regression_stump_model(adata, grn.copy())

    for col in ['cell_bool', 'threshold', 'pred_l', 'pred_r', 'cluster_bool_dt', 'weight']:
        assert col in out.columns


def test_prune_special_cases(simulate_data):
    adata, grn = simulate_data
    out = fit_regression_stump_model(adata, grn.copy())

    pruned, same_label = prune_special_cases(out)

    assert isinstance(pruned, pd.DataFrame)
    assert isinstance(same_label, pd.DataFrame)
    assert pruned.shape[0] <= grn.shape[0]


@pytest.mark.parametrize('mode', ['percent', 'quantile'])
def test_prune_wrt_n_cells(simulate_data, mode):
    adata, grn = simulate_data
    out = fit_regression_stump_model(adata, grn.copy())

    pruned = prune_wrt_n_cells(out, mode=mode, threshold=0.2)

    assert isinstance(pruned, pd.DataFrame)
    assert pruned.shape[0] <= grn.shape[0]


def test_calculate_weights_pipeline(simulate_data):
    adata, grn = simulate_data
    final = calculate_weights(adata, grn.copy())

    assert isinstance(final, pd.DataFrame)
    for col in ['cell_bool', 'threshold', 'pred_l', 'pred_r', 'cluster_bool_dt', 'weight']:
        assert col in final.columns
