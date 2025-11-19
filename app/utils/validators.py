"""
Input validation functions for the application.
"""

from typing import Tuple, Optional, Dict, List
from app.utils.constants import HANOI_BOUNDS
from app.core.exceptions import ValidationError
from app.core.logger import setup_app_logger

logger = setup_app_logger()


def validate_coordinates(lat: float, lon: float, bounds: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate coordinates are within bounds.
    
    Args:
        lat: Latitude
        lon: Longitude
        bounds: Optional bounds dict, defaults to Hanoi bounds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if bounds is None:
        bounds = HANOI_BOUNDS
    
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return False, "Coordinates must be numbers"
    
    if not (bounds['min_lat'] <= lat <= bounds['max_lat']):
        return False, f"Latitude must be between {bounds['min_lat']} and {bounds['max_lat']}"
    
    if not (bounds['min_lon'] <= lon <= bounds['max_lon']):
        return False, f"Longitude must be between {bounds['min_lon']} and {bounds['max_lon']}"
    
    return True, None


def validate_demand(demand: float) -> Tuple[bool, Optional[str]]:
    """
    Validate demand is a positive number.
    
    Args:
        demand: Customer demand
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(demand, (int, float)):
        return False, "Demand must be a number"
    
    if demand <= 0:
        return False, "Demand must be positive"
    
    if demand > 100000:
        return False, "Demand is unreasonably large (max 100000)"
    
    return True, None


def validate_capacity(capacity: float) -> Tuple[bool, Optional[str]]:
    """
    Validate vehicle capacity is a positive number.
    
    Args:
        capacity: Vehicle capacity
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(capacity, (int, float)):
        return False, "Vehicle capacity must be a number"
    
    if capacity <= 0:
        return False, "Vehicle capacity must be positive"
    
    return True, None


def validate_num_vehicles(num_vehicles: int) -> Tuple[bool, Optional[str]]:
    """
    Validate number of vehicles is a positive integer.
    
    Args:
        num_vehicles: Number of vehicles
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(num_vehicles, int):
        return False, "Number of vehicles must be an integer"
    
    if num_vehicles <= 0:
        return False, "Number of vehicles must be positive"
    
    return True, None


def validate_json_dataset(json_data: Dict, dataset_type: str = "hanoi_mockup") -> Tuple[bool, Optional[str], Optional[Dict]]:
    """
    Validate JSON dataset structure and content.
    
    Args:
        json_data: JSON data dictionary
        dataset_type: Type of dataset ("hanoi_mockup" or "solomon")
        
    Returns:
        Tuple of (is_valid, error_message, validated_data)
    """
    errors = []
    
    # Check required top-level keys
    required_keys = ['depot', 'customers']
    for key in required_keys:
        if key not in json_data:
            errors.append(f"Missing required field: {key}")
    
    if errors:
        return False, "; ".join(errors), None
    
    # Validate depot
    depot = json_data.get('depot', {})
    depot_required = ['id', 'x', 'y', 'demand']
    for key in depot_required:
        if key not in depot:
            errors.append(f"Depot missing required field: {key}")
    
    if 'x' in depot and 'y' in depot:
        if dataset_type == "hanoi_mockup":
            is_valid, error = validate_coordinates(depot['y'], depot['x'])  # y=lat, x=lon
            if not is_valid:
                errors.append(f"Depot coordinates invalid: {error}")
    
    if 'demand' in depot:
        is_valid, error = validate_demand(depot.get('demand', 0))
        if not is_valid and depot.get('demand', 0) != 0:
            errors.append(f"Depot demand invalid: {error}")
    
    # Validate customers
    customers = json_data.get('customers', [])
    if not isinstance(customers, list):
        errors.append("Customers must be a list")
    elif len(customers) == 0:
        errors.append("At least one customer is required")
    else:
        # Check for duplicate customer IDs
        customer_ids = [c.get('id') for c in customers if 'id' in c]
        if len(customer_ids) != len(set(customer_ids)):
            duplicate_ids = [id for id in customer_ids if customer_ids.count(id) > 1]
            errors.append(f"Duplicate customer IDs found: {list(set(duplicate_ids))}")
        
        # Check for duplicate coordinates (warn but don't block)
        seen_coords = set()
        duplicate_coords = []
        for customer in customers:
            if 'x' in customer and 'y' in customer:
                coord_key = (round(customer['x'], 6), round(customer['y'], 6))
                if coord_key in seen_coords:
                    duplicate_coords.append((customer.get('id', 'unknown'), customer['x'], customer['y']))
                seen_coords.add(coord_key)
        
        if duplicate_coords:
            logger.warning(f"Duplicate coordinates detected: {duplicate_coords}")
        
        for i, customer in enumerate(customers):
            customer_required = ['id', 'x', 'y', 'demand']
            for key in customer_required:
                if key not in customer:
                    errors.append(f"Customer {i+1} missing required field: {key}")
            
            if 'x' in customer and 'y' in customer:
                if dataset_type == "hanoi_mockup":
                    is_valid, error = validate_coordinates(customer['y'], customer['x'])
                    if not is_valid:
                        errors.append(f"Customer {i+1} coordinates invalid: {error}")
            
            if 'demand' in customer:
                is_valid, error = validate_demand(customer['demand'])
                if not is_valid:
                    errors.append(f"Customer {i+1} demand invalid: {error}")
    
    # Validate problem_config
    problem_config = json_data.get('problem_config', {})
    if 'vehicle_capacity' in problem_config:
        is_valid, error = validate_capacity(problem_config['vehicle_capacity'])
        if not is_valid:
            errors.append(f"Vehicle capacity invalid: {error}")
    else:
        errors.append("Missing vehicle_capacity in problem_config")
    
    if 'num_vehicles' in problem_config:
        is_valid, error = validate_num_vehicles(problem_config['num_vehicles'])
        if not is_valid:
            errors.append(f"Number of vehicles invalid: {error}")
    else:
        errors.append("Missing num_vehicles in problem_config")
    
    if errors:
        return False, "; ".join(errors), None
    
    # Return validated data in standard format
    validated_data = {
        'depot': depot,
        'customers': customers,
        'vehicle_capacity': problem_config['vehicle_capacity'],
        'num_vehicles': problem_config['num_vehicles'],
        'metadata': json_data.get('metadata', {}),
        'problem_config': problem_config
    }
    
    return True, None, validated_data

