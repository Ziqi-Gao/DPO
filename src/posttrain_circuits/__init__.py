"""Causal factorization experiments for post-training circuits.

The package root is deliberately metadata-only.  In particular, importing a
scheduler-neutral contract must not import the tensor or training stack before
the execution manifest and environment have been validated.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
