


import numpy as np
import pandas as pd
import scanpy as sc
import pytest
import random

@pytest.fixture
def simulate_data():
    """Fixture returning (adata, grn) dummy data for tests."""

    np.random.seed(42)
    random.seed(42)

    n_cells = 100
    n_genes = 200

    # Simulate count data
    x = np.random.randint(0, 100, size=(n_cells * n_genes, ))

    # Add sparsity
    n_zeros = int(n_cells * n_genes * 0.3)
    zero_indices = np.random.choice(n_cells * n_genes, n_zeros, replace=False)
    x[zero_indices] = 0

    # Create AnnData
    x = x.reshape((n_cells, n_genes))
    adata = sc.AnnData(x)
    tf_names = [f'TF{i}' for i in range(50)]
    non_tf_names = [f'Gene{i}' for i in range(150)]
    gene_names = tf_names + non_tf_names
    cell_names = [f'Cell{i}' for i in range(n_cells)]
    cell_anno = ['prog', ] * 50 + ['off', ] * 50
    adata.var_names = gene_names
    adata.obs_names_ = cell_names
    adata.obs['clusters'] = cell_anno

    # Generate random GRN
    n_edges = 100
    tfs = random.choices(tf_names, k=n_edges)
    targets = random.choices(gene_names, k=n_edges)

    grn = pd.DataFrame({
        'TF': tfs,
        'target': targets,
    })

    return adata, grn






