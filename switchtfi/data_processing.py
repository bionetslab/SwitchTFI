
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import scipy.sparse as sp

from scipy.stats import median_abs_deviation


def process_data(
        adata: sc.AnnData,
        qc_cells: bool = True,
        qc_genes: bool = True,
        magic_imputation: bool = True,
        verbosity: int = 0
):
    """
    End-to-end preprocessing pipeline for single-cell RNA-seq data.

    This function creates a working copy of the input AnnData object and applies
    a sequence of preprocessing steps, including cell QC, gene QC, normalization,
    log1p transformation, and MAGIC imputation.

    Args:
        adata (sc.AnnData): The input AnnData object with raw counts in `.X`.
        qc_cells (bool, optional): Whether to perform cell-level quality control.
            Defaults to True.
        qc_genes (bool, optional): Whether to filter lowly expressed genes.
            Defaults to True.
        magic_imputation (bool, optional): Whether to apply MAGIC imputation and
            store results in a new layer. Defaults to True.
        verbosity (int, optional): Verbosity level for logging. Defaults to 0.

    Returns:
        sc.AnnData: A new AnnData object with QC-filtered cells/genes,
        additional layers for normalized ('X_normalized'), log1p-transformed ('X_log1p_norm'),
        and MAGIC-imputed ('X_magic_imputed') data.
    """

    adata_work = adata.copy()

    if sp.issparse(adata_work.X):
        adata_work.X = adata_work.X.toarray()

    if qc_cells:
        adata_work = qc_cells_fct(adata=adata_work, verbosity=verbosity)

    if qc_genes:
        adata_work = qc_genes_fct(adata=adata_work, verbosity=verbosity)

    # Add layers with normalized and log-transformed data
    x_normalized = sc.pp.normalize_total(adata_work, target_sum=None, inplace=False)['X']
    adata_work.layers['X_normalized'] = x_normalized
    x_log1p_norm = sc.pp.log1p(x_normalized, copy=True)
    adata_work.layers['X_log1p_norm'] = x_log1p_norm

    if magic_imputation:
        adata_work = magic_imputation_fct(adata=adata_work, verbosity=verbosity)

    return adata_work


def qc_cells_fct(adata: sc.AnnData, verbosity: int = 1):
    """
    Perform cell-level quality control.

    This function flags mitochondrial genes, calculates QC metrics, identifies
    outliers based on median absolute deviation (MAD), and filters low-quality
    cells. Returns a filtered copy of the AnnData object.

    Args:
        adata (sc.AnnData): AnnData object with raw counts in `.X`.
        verbosity (int, optional): Verbosity level for logging. Defaults to 1.

    Returns:
        sc.AnnData: A new AnnData object with low-quality cells removed and QC
        metrics stored in `.obs`.
    """

    n_cells_before = adata.n_obs

    adata.var['mt'] = adata.var_names.str.startswith(('mt-', 'MT-'))

    if adata.var['mt'].any():
        qc_vars = ('mt', )
    else:
        qc_vars = tuple()
        if verbosity >= 1:
            print('Found no mitochondrial genes starting with "mt-" or "MT-".')

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        inplace=True,
        percent_top=[20],
        log1p=True,
    )

    adata.obs['outlier'] = (
            is_outlier(adata, obs_key_qc_metric='log1p_total_counts', nmads=5)
            | is_outlier(adata, obs_key_qc_metric='log1p_n_genes_by_counts', nmads=5)
            | is_outlier(adata, obs_key_qc_metric='pct_counts_in_top_20_genes', nmads=5)
    )

    if len(qc_vars) > 0:
        adata.obs['mt_outlier'] = (
                is_outlier(adata, obs_key_qc_metric='pct_counts_mt', nmads=3)
                | (adata.obs['pct_counts_mt'] > 8.0)
        )

    # Filter adata based on the identified outliers
    if len(qc_vars) > 0:
        mask = (~adata.obs.outlier) & (~adata.obs.mt_outlier)
    else:
        mask = (~adata.obs.outlier)

    adata_filtered = adata[mask].copy()

    if verbosity >= 1:
        print((
            f'# Number of cells before filtering: {n_cells_before}'
            f'\n# Number of cells after filtering: {adata_filtered.n_obs}'
            f'\n# Number of filtered cells: {n_cells_before - adata_filtered.n_obs}'
        ))

    return adata_filtered


def qc_genes_fct(adata: sc.AnnData, verbosity: int = 1):
    """
    Perform gene-level quality control.

    This function filters out genes that are expressed in fewer than 10 cells.
    Returns a filtered copy of the AnnData object.

    Args:
        adata (sc.AnnData): AnnData object with raw counts in `.X`.
        verbosity (int, optional): Verbosity level for logging. Defaults to 1.

    Returns:
        sc.AnnData: A new AnnData object with lowly expressed genes removed.
    """

    adata_work_qc_genes = adata.copy()

    n_genes_before = adata_work_qc_genes.n_vars

    sc.pp.filter_genes(adata_work_qc_genes, min_cells=10)

    if verbosity >= 1:
        print((
            f'# Number of genes before filtering: {n_genes_before}'
            f'\n# Number of genes after filtering: {adata_work_qc_genes.n_vars}'
            f'\n# Number of filtered genes with count less than 10: {n_genes_before - adata_work_qc_genes.n_vars}'
        ))

    return adata_work_qc_genes


def magic_imputation_fct(adata: sc.AnnData, verbosity: int = 1):
    """
    Apply MAGIC imputation to log1p-normalized data.

    This function, runs MAGIC, and stores the imputed result as a new
    layer ('X_magic_imputed') in the input AnnData object.

    Args:
        adata (sc.AnnData): AnnData object with normalized and log1p-transformed
            data in `adata.layers['X_log1p_norm']`.
        verbosity (int, optional): Verbosity level for MAGIC output.
            Defaults to 1.

    Returns:
        sc.AnnData: The same AnnData object with an additional
        `adata.layers['X_magic_imputed']` containing imputed values.
    """

    # Create dummy AnnData
    adata_work_magic = sc.AnnData(
        X=adata.layers['X_log1p_norm'],
        obs=pd.DataFrame(index=adata.obs_names),
        var=pd.DataFrame(index=adata.var_names),
    )

    # Run MAGIC with default parameter and diffusion time t
    sce.pp.magic(
        adata=adata_work_magic,
        name_list='all_genes',
        knn=5,
        decay=1,
        knn_max=None,
        t=1,
        n_pca=100,
        solver='exact',
        random_state=None,
        n_jobs=None,
        verbose=(verbosity >= 1),
        copy=None
    )

    adata.layers['X_magic_imputed'] = adata_work_magic.X.copy()

    return adata


def is_outlier(
        adata: sc.AnnData,
        obs_key_qc_metric: str,
        nmads: int = 5
) -> pd.Series:
    """
    Identify outliers in a QC metric using Median Absolute Deviations, MAD = median(|x_i - median(x)|).

    This function flags outliers in a given observation key (`obs_key_qc_metric`) of an AnnData object
    based on a threshold determined by the number of median absolute deviations (MAD) from the median.

    Args:
        adata (sc.AnnData): The AnnData object containing the QC metric to analyze.
        obs_key_qc_metric (str): The key in `adata.obs` corresponding to the QC metric to check for outliers.
        nmads (int, optional): The number of MADs from the median to define an outlier. Defaults to 5.

    Returns:
        pd.Series: A boolean series where True indicates an outlier.
    """
    # Uses Median Absolute Deviations as outlier criterion for QC-metric x_i, x
    # MAD = median(|x_i - median(x)|)
    m = adata.obs[obs_key_qc_metric]

    median = np.median(m)
    mad = median_abs_deviation(m)

    lower_bound = median - nmads * mad
    upper_bound = median + nmads * mad

    outlier_bool = (m < lower_bound) | (m > upper_bound)

    return outlier_bool





