"""Public package API for microio."""

import logging

from .reader.open import open_dataset

logging.getLogger("microio").addHandler(logging.NullHandler())

__all__ = ["open_dataset"]
