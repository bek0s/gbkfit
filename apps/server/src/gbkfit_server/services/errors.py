"""
The errors of the requests to the services, which the API answers with
their responses (e.g. not found).
"""


__all__ = [
    'Cancelled',
    'Failed',
    'Invalid',
    'NotFound',
    'ServiceError'
]


class ServiceError(Exception):
    """An error of a request to a service."""


class NotFound(ServiceError):
    """What a request names does not exist."""


class Invalid(ServiceError):
    """A request is invalid (e.g. a path out of the workspace)."""


class Failed(ServiceError):
    """A service failed (e.g. the worker of a session ended)."""


class Cancelled(ServiceError):
    """What a request asked for was cancelled (e.g. a render)."""
