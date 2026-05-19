#!/usr/bin/env python3
# Legacy/specialized entry point. Current production report is reports/structural_fg_svm_kronecker_report.py with --report-style product_v1.
"""Evidence-first robustness report (alias entry point)."""

from reports.structural_fg_svm_robustness_report import main

if __name__ == "__main__":
    raise SystemExit(main())
