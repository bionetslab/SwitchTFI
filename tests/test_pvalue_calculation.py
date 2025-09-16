
import numpy as np
import pytest

from switchtfi.pvalue_calculation import (
    weights_to_w_y_adjusted_pvalue,
    weights_to_emp_pvals,
    compute_westfall_young_adjusted_pvalues,
    compute_empirical_pvalues,
    adjust_pvals,
    compute_corrected_pvalues,
    remove_insignificant_edges,
)


# -------------------
# Auxiliary function tests
# -------------------

def test_weights_to_w_y_adjusted_pvalue():
    true_weights = np.array([0.25, 0.9])
    permutation_weights = np.array(
        [
            [0.25, 1.0, 0.25],
            [1.0, 0.25, 0.3]
        ]
    )

    # Per permutation argmaxs = [1.0, 1.0, 0.3] -> per edge p-values: [0, 2 / 3]

    pvals = weights_to_w_y_adjusted_pvalue(true_weights, permutation_weights)
    assert pvals.shape == (2,)
    assert np.all((0 <= pvals) & (pvals <= 1))
    assert pvals[0] == 1
    assert pvals[1] == 2 / 3


def test_test_statistic_to_emp_pvals_exact_and_nonexact():
    true_weights = np.array([1.0, 0.25, 0.3])
    permutation_weights = np.array(
        [
            [0.25, 0.25, 0.25],
            [1.0, 1.0, 1.0],
            [0.25, 0.25, 9.0]
        ]
    )

    pvals_exact = weights_to_emp_pvals(true_weights, permutation_weights, exact_pval=True)
    pvals_nonexact = weights_to_emp_pvals(true_weights, permutation_weights, exact_pval=False)

    assert pvals_exact.shape == (3,)
    assert pvals_nonexact.shape == (3,)
    assert np.all((0 <= pvals_exact) & (pvals_exact <= 1))
    assert np.all((0 <= pvals_nonexact) & (pvals_nonexact <= 1))
    assert pvals_exact[0] == 1 / 4 and pvals_exact[1] == 1 and pvals_exact[2] == 2 / 4
    assert pvals_nonexact[0] == 0 and pvals_nonexact[1] == 1 and pvals_nonexact[2] == 1 / 3


# -------------------
# Integration tests with simulate_data
# -------------------

@pytest.mark.slow
def test_compute_westfall_young_adjusted_pvalues(simulate_data):
    adata, grn = simulate_data

    out = compute_westfall_young_adjusted_pvalues(
        adata, grn.copy(), n_permutations=5
    )
    assert 'pvals_wy' in out.columns
    assert out['pvals_wy'].between(0, 1).all()


@pytest.mark.slow
def test_compute_empirical_pvalues(simulate_data):
    adata, grn = simulate_data
    out = compute_empirical_pvalues(
        adata, grn.copy(), n_permutations=5
    )
    assert 'emp_pvals' in out.columns
    assert out['emp_pvals'].between(0, 1).all()


@pytest.mark.parametrize('method', ['bonferroni', 'sidak', 'fdr_bh', 'fdr_by'])
def test_adjust_pvals(simulate_data, method):
    _, grn = simulate_data
    grn = grn.copy()
    grn['emp_pvals'] = [0.01, ] * 20 + [0.5] * (grn.shape[0] - 20)

    out = adjust_pvals(grn, pval_key='emp_pvals', method=method)
    assert f'pvals_{method}' in out.columns
    assert out[f'pvals_{method}'].between(0, 1).all()
    if method in ['bonferroni', 'sidak']:
        assert (out[f'pvals_{method}'] >= grn['emp_pvals']).all()


@pytest.mark.parametrize('method', ['wy', 'bonferroni', 'sidak', 'fdr_bh', 'fdr_by'])
def test_compute_corrected_pvalues_methods(simulate_data, method):
    adata, grn = simulate_data
    out = compute_corrected_pvalues(
        adata, grn.copy(),
        method=method,
        n_permutations=5
    )
    assert f'pvals_{method}' in out.columns
    assert out[f'pvals_{method}'].between(0, 1).all()


def test_remove_insignificant_edges(simulate_data):
    _, grn = simulate_data
    grn = grn.copy()
    grn['pvals_wy'] = [0.01, ] * 20 + [0.2] * (grn.shape[0] - 20)
    out = remove_insignificant_edges(grn, alpha=0.05, p_value_key='pvals_wy')

    assert (out['pvals_wy'] <= 0.05).all()
    assert out.shape[0] == 20
