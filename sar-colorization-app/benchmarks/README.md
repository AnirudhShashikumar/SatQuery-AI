# SatQuery benchmark data

This directory is the canonical, versioned reporting source for the SatQuery benchmark module. It is deliberately separate from inference code and generated application caches.

- `schema/` defines strict benchmark, suite, and demo contracts.
- `results/` contains normalized records whose values are copied from named source artifacts.
- `suites/` selects preferred records and records coverage gaps.
- `demos/` contains cautious, attributed demonstration metadata.

Run `python scripts/import_existing_benchmarks.py`, then `python scripts/validate_benchmark_results.py` to rebuild and validate the repository data. Importing never runs inference and never infers an absent value. Missing measurements remain `null`; smoke and operational records retain their non-quality status.
