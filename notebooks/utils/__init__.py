"""Reusable helpers for benchmark analysis notebooks."""

from .constraints import (
    DEFAULT_OHE_VALID_TOL,
    constraint_quality_rows,
    constraint_quality_summary,
    constraint_quality_tables,
)
from .filters import (
    select_methods,
    shared_success_retention_summary,
    shared_success_subset,
    success_only,
)
from .io import feature_columns, load_result, load_result_set, normalize_benchmark_df
from .mnist import mnist_task_coverage_table, mnist_validity_by_target_table
from .plots import (
    plot_failure_stacked_bar,
    plot_feature_change_heatmap,
    plot_heatmap,
    plot_metric_boxplots,
    plot_runtime_bars,
    plot_runtime_boxplot,
    plot_validity_bar,
    plot_validity_distance_curves,
)
from .robustness import (
    evaluate_empirical_robustness_curves,
    load_tabular_benchmark_models,
    numerical_feature_indices,
    sample_lp_ball,
)
from .style import DEFAULT_METHOD_LABELS, method_palette, setup_notebook_style
from .style import bold_best_values, style_method_table
from .summaries import (
    benchmark_overview,
    build_validity_curve_df,
    curve_endpoint_table,
    curve_auc_table,
    failure_summary,
    feature_change_summary,
    method_order,
    metric_table,
    proximity_summary,
    runtime_summary,
    validity_closeness_score,
    validity_proximity_curve,
    validity_summary,
)

__all__ = [
    "DEFAULT_METHOD_LABELS",
    "DEFAULT_OHE_VALID_TOL",
    "benchmark_overview",
    "bold_best_values",
    "build_validity_curve_df",
    "constraint_quality_rows",
    "constraint_quality_summary",
    "constraint_quality_tables",
    "curve_endpoint_table",
    "curve_auc_table",
    "evaluate_empirical_robustness_curves",
    "failure_summary",
    "feature_change_summary",
    "feature_columns",
    "load_result",
    "load_result_set",
    "method_order",
    "method_palette",
    "metric_table",
    "load_tabular_benchmark_models",
    "mnist_task_coverage_table",
    "mnist_validity_by_target_table",
    "normalize_benchmark_df",
    "numerical_feature_indices",
    "plot_failure_stacked_bar",
    "plot_feature_change_heatmap",
    "plot_heatmap",
    "plot_metric_boxplots",
    "plot_runtime_bars",
    "plot_runtime_boxplot",
    "plot_validity_bar",
    "plot_validity_distance_curves",
    "proximity_summary",
    "runtime_summary",
    "sample_lp_ball",
    "select_methods",
    "setup_notebook_style",
    "style_method_table",
    "shared_success_retention_summary",
    "shared_success_subset",
    "success_only",
    "validity_closeness_score",
    "validity_proximity_curve",
    "validity_summary",
]
