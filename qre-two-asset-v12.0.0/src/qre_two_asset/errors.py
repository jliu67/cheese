"""Domain-specific failures with fail-closed semantics."""


class QREError(RuntimeError):
    """Base error for the package."""


class ConfigError(QREError):
    """Configuration is incomplete or internally inconsistent."""


class DataIntegrityError(QREError):
    """Input data violate the documented contract."""


class LookAheadError(DataIntegrityError):
    """Information unavailable on the decision date was used."""


class HoldoutError(QREError):
    """The final holdout protocol was violated."""


class DeploymentGateError(QREError):
    """A live allocation was requested from an unapproved research result."""
