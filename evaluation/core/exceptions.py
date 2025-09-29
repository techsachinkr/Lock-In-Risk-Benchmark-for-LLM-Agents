"""
Custom exceptions for lock-in evaluation framework
"""


class LockInEvalException(Exception):
    """Base exception for lock-in evaluation framework"""
    pass


class ScenarioException(LockInEvalException):
    """Exception raised in scenario execution"""
    pass


class MetricsException(LockInEvalException):
    """Exception raised in metrics calculation"""
    pass


class ConfigurationException(LockInEvalException):
    """Exception raised for configuration issues"""
    pass


class ValidationException(LockInEvalException):
    """Exception raised for validation failures"""
    pass