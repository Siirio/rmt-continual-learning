"""Deterministic results, tables, figures, and report generation."""

from .example_data import generate_example_package
from .tables import aggregate_method_metrics, markdown_table

__all__ = ["aggregate_method_metrics", "generate_example_package", "markdown_table"]

