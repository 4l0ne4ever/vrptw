"""
Custom exceptions for VRP-GA System.
Provides specific exception classes for different error types.
"""


class VRPException(Exception):
    """Base exception for VRP-GA system."""
    
    def __init__(self, message: str = "", details: dict = None):
        """
        Initialize VRP exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class InfeasibleSolutionError(VRPException):
    """Raised when solution violates constraints."""
    pass


class CapacityViolationError(VRPException):
    """Raised when vehicle capacity is exceeded."""
    
    def __init__(self, route_id: int = None, current_load: float = None, 
                 capacity: float = None, customer_id: int = None):
        """
        Initialize capacity violation error.
        
        Args:
            route_id: Route ID with violation
            current_load: Current load in route
            capacity: Vehicle capacity limit
            customer_id: Customer ID causing violation
        """
        message = "Vehicle capacity constraint violated"
        details = {}
        
        if route_id is not None:
            details['route_id'] = route_id
        if current_load is not None:
            details['current_load'] = current_load
        if capacity is not None:
            details['capacity'] = capacity
        if customer_id is not None:
            details['customer_id'] = customer_id
        
        if details:
            message += f": Route {route_id} has load {current_load} > capacity {capacity}"
        
        super().__init__(message, details)


class TimeWindowViolationError(VRPException):
    """Raised when time window constraint is violated."""
    
    def __init__(self, customer_id: int = None, arrival_time: float = None,
                 ready_time: float = None, due_date: float = None):
        """
        Initialize time window violation error.
        
        Args:
            customer_id: Customer ID with violation
            arrival_time: Actual arrival time
            ready_time: Earliest allowed arrival time
            due_date: Latest allowed arrival time
        """
        message = "Time window constraint violated"
        details = {}
        
        if customer_id is not None:
            details['customer_id'] = customer_id
        if arrival_time is not None:
            details['arrival_time'] = arrival_time
        if ready_time is not None:
            details['ready_time'] = ready_time
        if due_date is not None:
            details['due_date'] = due_date
        
        if details:
            if arrival_time < ready_time:
                message += f": Customer {customer_id} arrived too early ({arrival_time} < {ready_time})"
            elif arrival_time > due_date:
                message += f": Customer {customer_id} arrived too late ({arrival_time} > {due_date})"
        
        super().__init__(message, details)


class DistanceCalculationError(VRPException):
    """Raised when distance calculation fails."""
    
    def __init__(self, from_id: int = None, to_id: int = None, reason: str = None):
        """
        Initialize distance calculation error.
        
        Args:
            from_id: Source node ID
            to_id: Destination node ID
            reason: Reason for failure
        """
        message = "Distance calculation failed"
        details = {}
        
        if from_id is not None:
            details['from_id'] = from_id
        if to_id is not None:
            details['to_id'] = to_id
        if reason:
            details['reason'] = reason
        
        if details:
            message += f": Cannot calculate distance from {from_id} to {to_id}"
            if reason:
                message += f" ({reason})"
        
        super().__init__(message, details)


class DatasetNotFoundError(VRPException):
    """Raised when dataset is not found."""
    
    def __init__(self, dataset_name: str = None, dataset_type: str = None):
        """
        Initialize dataset not found error.
        
        Args:
            dataset_name: Name of the missing dataset
            dataset_type: Type of dataset (solomon/mockup)
        """
        message = "Dataset not found"
        details = {}
        
        if dataset_name:
            details['dataset_name'] = dataset_name
        if dataset_type:
            details['dataset_type'] = dataset_type
        
        if dataset_name:
            message += f": '{dataset_name}'"
            if dataset_type:
                message += f" (type: {dataset_type})"
        
        super().__init__(message, details)


class InvalidConfigurationError(VRPException):
    """Raised when configuration parameters are invalid."""
    
    def __init__(self, parameter: str = None, value: any = None, 
                 expected: str = None):
        """
        Initialize invalid configuration error.
        
        Args:
            parameter: Parameter name
            value: Invalid value
            expected: Expected value or range
        """
        message = "Invalid configuration parameter"
        details = {}
        
        if parameter:
            details['parameter'] = parameter
        if value is not None:
            details['value'] = value
        if expected:
            details['expected'] = expected
        
        if parameter:
            message += f": {parameter} = {value}"
            if expected:
                message += f" (expected: {expected})"
        
        super().__init__(message, details)


class DecodingError(VRPException):
    """Raised when chromosome decoding fails."""
    
    def __init__(self, chromosome: list = None, reason: str = None):
        """
        Initialize decoding error.
        
        Args:
            chromosome: Chromosome that failed to decode
            reason: Reason for failure
        """
        message = "Chromosome decoding failed"
        details = {}
        
        if chromosome is not None:
            details['chromosome_length'] = len(chromosome) if chromosome else 0
        if reason:
            details['reason'] = reason
        
        if reason:
            message += f": {reason}"
        
        super().__init__(message, details)

