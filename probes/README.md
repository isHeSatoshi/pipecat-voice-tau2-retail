# Probes and diagnostic utilities

This folder keeps experimental probes, one-off analysis helpers, and captured diagnostic logs out of the product source tree.

- `analysis/` contains saved-run analysis and reward backfill utilities.
- `pipeline/` contains small pipeline probes and smoke utilities.
- `tests/` contains exploratory full-pipeline probes that are not part of the regular test suite.
- `logs/` contains retained diagnostic output from earlier runs.

These files are preserved because they document how the system was debugged. They are not required for importing or running the main package.
