"""Errors raised at the OPD/ServerScheduler trust boundary."""


class AdapterError(RuntimeError):
    """Base class for adapter failures."""


class AdapterValidationError(AdapterError, ValueError):
    """Untrusted scheduler input failed closed validation."""


class DispatchError(AdapterError):
    """A validated unit could not be dispatched or completed safely."""
