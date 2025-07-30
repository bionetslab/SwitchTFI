

def main():

    import numpy as np
    import matplotlib.pyplot as plt

    from switchtfi import (
        erythrocytes, preendocrine_beta, preendocrine_alpha,
        erythrocytes_grn, preendocrine_beta_grn, preendocrine_alpha_grn,
        fit_model,
        calculate_weights,
        compute_corrected_pvalues,
        remove_insignificant_edges,
        rank_tfs,
        plot_regulon
    )

    tissue = 'ery'  # 'ery', 'beta', 'alpha'

    # Load the preprocessed scRNA-seq data and GRN (previously inferred with the Scenic method)
    if tissue == 'ery':
        data = erythrocytes()
        grn = erythrocytes_grn()
        prog_off_annotations_key = 'prog_off'
    elif tissue == 'beta':
        data = preendocrine_beta()
        grn = preendocrine_beta_grn()
        prog_off_annotations_key = 'clusters'
    else:
        data = preendocrine_alpha()
        grn = preendocrine_alpha_grn()
        prog_off_annotations_key = 'clusters'

    # ### Perform SwitchTFI analyses ### #
    # - Compute weights and empirical corrected p-values for each edge in the input GRN
    # - Prune the input GRN: remove edge if p-value > FWER-threshold  ==> transition GRN
    # - Rank transcription factors according to centrality in transition GRN

    # Run fit function => transition GRN and ranked TFs
    np.random.seed(42)
    transition_grn, ranked_tfs = fit_model(
        adata=data,
        grn=grn,
        layer_key='magic_imputed',
        n_permutations=1000,
        clustering_obs_key=prog_off_annotations_key,
    )

    print(f'# ### Transition GRN:\n{transition_grn}')
    print(f'# ### Transition driver TFs (ranked by PageRank):\n{ranked_tfs}')

    # ### The steps of SwitchTFI's analyses can be performed individually as well
    # Note: Now the weighted outdegree is used for ranking the TFs in the transition GRN
    np.random.seed(42)
    grn_weighted = calculate_weights(
        adata=data,
        grn=grn,
        layer_key='magic_imputed',
        clustering_obs_key=prog_off_annotations_key
    )

    grn_weighted_with_pvals = compute_corrected_pvalues(
        adata=data,
        grn=grn_weighted,
        n_permutations=1000,
        clustering_obs_key=prog_off_annotations_key,
    )

    transition_grn = remove_insignificant_edges(grn=grn_weighted_with_pvals, alpha=0.05, inplace=False)

    ranked_tfs = rank_tfs(
        grn=transition_grn,
        reverse=False,
        centrality_measure='out_degree',
        weight_key='score'
    )

    # Subset the GRN to the relevant columns
    transition_grn = transition_grn[['TF', 'target', 'weight', 'pvals_wy']]

    print(f'# ### Transition GRN:\n{transition_grn}')
    print(f'# ### Transition driver TFs (ranked by weighted outdegree):\n{ranked_tfs}')

    # Visualize the regulon of the top ranked TF
    top_tf = ranked_tfs.loc[0, 'gene']

    plot_regulon(
        grn=transition_grn,
        tf=top_tf,
        sort_by='score',
        top_k=10
    )

    plt.show()


if __name__ == '__main__':

    main()

    print('done')
