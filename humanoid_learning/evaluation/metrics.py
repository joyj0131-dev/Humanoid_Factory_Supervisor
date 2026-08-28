"""Metric formatting shared by evaluation reports.

Success rate must always show both the percentage and the raw fraction
(e.g. ``82% (41/50)``) -- see PROJECT_CONTEXT.md, Evaluation.
"""

from __future__ import annotations


def format_success_rate(n_success: int, n_total: int) -> str:
    if n_total == 0:
        return "n/a (0/0)"
    pct = 100.0 * n_success / n_total
    return f"{pct:.0f}% ({n_success}/{n_total})"
