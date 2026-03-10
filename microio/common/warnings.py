"""Custom warnings emitted by microio conversion logic."""


class MicroioWarning(UserWarning):
    """Base warning for microio."""


class NotImplementedMetadataWarning(MicroioWarning):
    """Raised when metadata cannot be inferred and a fallback is used."""
