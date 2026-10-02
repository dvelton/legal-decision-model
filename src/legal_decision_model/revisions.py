"""Helpers for loading the exact reviewed Laya checkpoint."""

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def pinned_laya_revision(revision: str) -> Iterator[None]:
    """Pin Laya's revision and omit calibration warnings unused by the task head."""
    previous = os.environ.get("LAYA_REVISION")
    os.environ["LAYA_REVISION"] = revision
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"laya: this checkpoint ships invalid temperatures.*",
                category=RuntimeWarning,
            )
            yield
    finally:
        if previous is None:
            os.environ.pop("LAYA_REVISION", None)
        else:
            os.environ["LAYA_REVISION"] = previous
