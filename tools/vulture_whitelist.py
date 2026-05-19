"""
Names intentionally referenced dynamically (CLI, joblib, HTML callbacks).
Used with: vulture . --make-whitelist  (then merge) or manual review.

SpectraReason: do not delete items listed here without grep + test coverage.
"""

# CLI / report entry points
main = True  # noqa: F821
batch = True  # noqa: F821
run_batch = True  # noqa: F821

# sklearn / joblib persisted attributes
classes_ = True  # noqa: F821
n_features_in_ = True  # noqa: F821
feature_names_in_ = True  # noqa: F821

# Plotly / HTML report hooks
build_front_card_stack = True  # noqa: F821
write_interactive_report_html = True  # noqa: F821
build_product_tables_stack = True  # noqa: F821

# Guardrail / rules dispatch by string mode
apply_v3_guardrails = True  # noqa: F821
assign_functional_groups_from_evidence = True  # noqa: F821

# Backward-compatible re-exports (ml.fg_label_configs)
LEGACY_FG_RULES = True  # noqa: F821
infer_legacy_fg_vector = True  # noqa: F821

# Report metadata parameters (wired from CLI; may be used in future meta block)
rules_preset_label = True  # noqa: F821
rules_config_path_label = True  # noqa: F821

# Static export entry (imported inside run_batch branch)
export_static_figure_bundle = True  # noqa: F821

# Training diagnostics API surface
recall = True  # noqa: F821
