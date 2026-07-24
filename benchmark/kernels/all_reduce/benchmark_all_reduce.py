"""Alternate entry point for the SGLang all-reduce tuning benchmark.

The maintained implementation and CLI live in ``benchmark_torch_symm_mem.py``.
This preserves the entry-point filename, but its CLI and behavior now match
that benchmark rather than the historical two-backend implementation.
"""

if __package__:
    from .benchmark_torch_symm_mem import main
else:
    # Direct script execution places this directory, rather than its parent
    # package, on sys.path.
    from benchmark_torch_symm_mem import main


if __name__ == "__main__":
    raise SystemExit(main())
