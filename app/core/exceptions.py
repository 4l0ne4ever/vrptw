"""
Custom exceptions for web application.
"""


class AppException(Exception):
    """Base exception for application errors."""
    pass


class DatabaseError(AppException):
    """Database operation errors."""
    pass


class ValidationError(AppException):
    """Data validation errors."""
    pass


class DatasetError(AppException):
    """Dataset-related errors."""
    pass


class OptimizationError(AppException):
    """Optimization process errors."""
    pass


class ExportError(AppException):
    """Export operation errors."""
    pass

