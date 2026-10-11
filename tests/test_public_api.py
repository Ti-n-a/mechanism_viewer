import mechanism_viewer


def test_expected_public_api_is_exposed():
    expected_names = {
        "ColType",
        "DEFAULT_RANDOM_STATE",
        "generate_synthetic_dataset",
        "apply_mcar",
        "apply_mar",
        "apply_mnar",
        "apply_missingness",
        "generate_dataset_with_missingness",
        "plot_total_missingness",
        "plot_missing_rate",
        "plot_missingness_distribution",
        "missing_rows_matrix",
        "missing_rate_matrix",
        "missingness_misscol_corr",
        "value_misscol_corr",
        "complete_and_misscol_corr",
        "misscol_vs_all_corr",
        "upset_missing_rows",
        "missingness_dendrogram",
        "scatter_missingness_comparison",
        "scatter_missingness_comparison_line",
        "boxplot_comparison",
        "parallel_comparison",
        "scatter_imputation_comparison",
        "plot_imputation_distribution",
        "run_random_forest",
        "run_logistic_regression",
        "test_mar_from_model_accuracy",
        "interpret_mar_abd",
        "little_mcar_global",
        "interpret_mcar_p_value",
        "little_mcar_pairwise",
        "plot_mcar_pairwise",
    }

    missing = sorted(name for name in expected_names if not hasattr(mechanism_viewer, name))
    assert missing == []
