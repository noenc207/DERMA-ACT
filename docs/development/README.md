# Development Documentation (`docs/development/`)

- **Code Quality & Test Standards:** Formatting, linting, and `pytest` settings are configured in [`../../pyproject.toml`](../../pyproject.toml). See [`../../tests/README.md`](../../tests/README.md) for clean-room verification commands.
- **Notebook Hygiene:** Public notebooks under [`../../notebooks/`](../../notebooks/README.md) are validated with `nbformat`, executed via `nbclient`, and committed with stripped outputs (`execution_count: null`) and portable relative paths.
- **Related:** [Documentation Index](../README.md) · [Reproducibility](../../provenance/REPRODUCIBILITY.md)
