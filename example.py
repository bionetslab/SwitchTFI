
from switchtfi.data import erythrocytes, erythrocytes_grn
from switchtfi.fit import fit_model
from switchtfi.weight_fitting import calculate_weights
from switchtfi.pvalue_calculation import compute_westfall_young_adjusted_pvalues, remove_insignificant_edges
from switchtfi.tf_ranking import rank_tfs


def main():
    # Load the preprocessed scRNA-seq data
    erydata = erythrocytes()

    # Load the GRN (previously inferred with the Scenic method)
    erygrn = erythrocytes_grn()

    # ### Perform SwitchTFI analyses
    # - Compute weights and empirical corrected p-values for each edge in the input GRN
    # - Prune the input GRN: remove edge if p-value > FWER-threshold  ==> transition GRN
    # - Rank transcription factors according to centrality in transition GRN

    # Run fit function => transition GRN and ranked TFs
    ery_transitiongrn, ery_ranked_tfs = fit_model(adata=erydata,
                                                  grn=erygrn,
                                                  clustering_obs_key='prog_off',
                                                  verbosity=1)
    print(f'# ### Transition GRN erythrocytes differentiation:\n{ery_transitiongrn}')
    print(f'# ### Transition driver TFs erythrocytes differentiation:\n{ery_ranked_tfs.head(10)}')

    # The steps of SwitchTFI's analyses can be performed individually as well
    # Note that now the weighted outdegree is used for ranking the TFs in the transition GRN
    erygrn_weighted = calculate_weights(adata=erydata,
                                        grn=erygrn,
                                        layer_key='magic_imputed',
                                        clustering_obs_key='prog_off')

    erygrn_weighted_with_pvals = compute_westfall_young_adjusted_pvalues(adata=erydata,
                                                                         grn=erygrn_weighted,
                                                                         n_permutations=1000,
                                                                         clustering_obs_key='prog_off')

    ery_transition_grn = remove_insignificant_edges(grn=erygrn_weighted_with_pvals, alpha=0.05, inplace=False)

    ery_ranked_tfs = rank_tfs(grn=ery_transition_grn,
                              reverse=False,
                              centrality_measure='out_degree',
                              weight_key='score')

    print(f'# ### Transition GRN erythrocytes differentiation:\n{ery_transitiongrn}')
    print(f'# ### Transition driver TFs erythrocytes differentiation:\n{ery_ranked_tfs.head(10)}')


if __name__ == '__main__':

    main()

    print('done')
