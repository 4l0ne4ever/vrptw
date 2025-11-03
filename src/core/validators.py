"""
Validation layer for VRP-GA System.
Provides validators for configuration and data validation.
"""

from typing import Dict, List, Optional
from src.models.vrp_model import VRPProblem
from src.core.exceptions import InvalidConfigurationError


class ConfigValidator:
    """Validate configuration parameters."""
    
    @staticmethod
    def validate_ga_config(config: Dict) -> bool:
        """
        Validate GA configuration.
        
        Args:
            config: GA configuration dictionary
            
        Returns:
            True if valid, raises InvalidConfigurationError otherwise
            
        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        required_keys = [
            'population_size',
            'generations',
            'crossover_prob',
            'mutation_prob',
            'tournament_size',
            'elitism_rate'
        ]
        
        # Check required keys
        for key in required_keys:
            if key not in config:
                raise InvalidConfigurationError(
                    parameter=key,
                    value=None,
                    expected="Required parameter"
                )
        
        # Validate population_size
        if config['population_size'] < 10:
            raise InvalidConfigurationError(
                parameter='population_size',
                value=config['population_size'],
                expected=">= 10"
            )
        
        if config['population_size'] > 1000:
            raise InvalidConfigurationError(
                parameter='population_size',
                value=config['population_size'],
                expected="<= 1000"
            )
        
        # Validate generations
        if config['generations'] < 1:
            raise InvalidConfigurationError(
                parameter='generations',
                value=config['generations'],
                expected=">= 1"
            )
        
        # Validate probabilities
        if not 0 <= config['crossover_prob'] <= 1:
            raise InvalidConfigurationError(
                parameter='crossover_prob',
                value=config['crossover_prob'],
                expected="[0, 1]"
            )
        
        if not 0 <= config['mutation_prob'] <= 1:
            raise InvalidConfigurationError(
                parameter='mutation_prob',
                value=config['mutation_prob'],
                expected="[0, 1]"
            )
        
        if not 0 <= config['elitism_rate'] <= 1:
            raise InvalidConfigurationError(
                parameter='elitism_rate',
                value=config['elitism_rate'],
                expected="[0, 1]"
            )
        
        # Validate tournament_size
        if config['tournament_size'] < 2:
            raise InvalidConfigurationError(
                parameter='tournament_size',
                value=config['tournament_size'],
                expected=">= 2"
            )
        
        if config['tournament_size'] > config.get('population_size', 100):
            raise InvalidConfigurationError(
                parameter='tournament_size',
                value=config['tournament_size'],
                expected=f"<= population_size ({config.get('population_size', 100)})"
            )
        
        return True
    
    @staticmethod
    def validate_vrp_config(config: Dict) -> bool:
        """
        Validate VRP configuration.
        
        Args:
            config: VRP configuration dictionary
            
        Returns:
            True if valid, raises InvalidConfigurationError otherwise
            
        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        # Validate vehicle_capacity
        if 'vehicle_capacity' in config:
            if config['vehicle_capacity'] <= 0:
                raise InvalidConfigurationError(
                    parameter='vehicle_capacity',
                    value=config['vehicle_capacity'],
                    expected="> 0"
                )
        
        # Validate num_vehicles
        if 'num_vehicles' in config:
            if config['num_vehicles'] <= 0:
                raise InvalidConfigurationError(
                    parameter='num_vehicles',
                    value=config['num_vehicles'],
                    expected="> 0"
                )
        
        # Validate traffic_factor
        if 'traffic_factor' in config:
            if config['traffic_factor'] <= 0:
                raise InvalidConfigurationError(
                    parameter='traffic_factor',
                    value=config['traffic_factor'],
                    expected="> 0"
                )
        
        # Validate cod_fee_rate
        if 'cod_fee_rate' in config:
            if not 0 <= config['cod_fee_rate'] <= 1:
                raise InvalidConfigurationError(
                    parameter='cod_fee_rate',
                    value=config['cod_fee_rate'],
                    expected="[0, 1]"
                )
        
        return True
    
    @staticmethod
    def validate_mockup_config(config: Dict) -> bool:
        """
        Validate mockup data generation configuration.
        
        Args:
            config: Mockup configuration dictionary
            
        Returns:
            True if valid, raises InvalidConfigurationError otherwise
            
        Raises:
            InvalidConfigurationError: If configuration is invalid
        """
        # Validate n_customers
        if 'n_customers' in config:
            if config['n_customers'] < 1:
                raise InvalidConfigurationError(
                    parameter='n_customers',
                    value=config['n_customers'],
                    expected=">= 1"
                )
        
        # Validate demand range
        if 'demand_min' in config and 'demand_max' in config:
            if config['demand_min'] < 0:
                raise InvalidConfigurationError(
                    parameter='demand_min',
                    value=config['demand_min'],
                    expected=">= 0"
                )
            
            if config['demand_max'] < config['demand_min']:
                raise InvalidConfigurationError(
                    parameter='demand_max',
                    value=config['demand_max'],
                    expected=f">= demand_min ({config['demand_min']})"
                )
        
        # Validate service_time
        if 'service_time' in config:
            if config['service_time'] < 0:
                raise InvalidConfigurationError(
                    parameter='service_time',
                    value=config['service_time'],
                    expected=">= 0"
                )
        
        # Validate clustering method
        if 'clustering' in config:
            valid_methods = ['random', 'kmeans', 'radial']
            if config['clustering'] not in valid_methods:
                raise InvalidConfigurationError(
                    parameter='clustering',
                    value=config['clustering'],
                    expected=f"One of {valid_methods}"
                )
        
        return True


class DataValidator:
    """Validate VRP data and problem instances."""
    
    @staticmethod
    def validate_problem(problem: VRPProblem) -> bool:
        """
        Validate VRP problem instance.
        
        Args:
            problem: VRP problem instance
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If problem is invalid
        """
        if problem.vehicle_capacity <= 0:
            raise ValueError("Vehicle capacity must be positive")
        
        if problem.num_vehicles <= 0:
            raise ValueError("Number of vehicles must be positive")
        
        if len(problem.customers) == 0:
            raise ValueError("No customers in problem")
        
        # Check if any customer demand exceeds vehicle capacity
        max_demand = max(c.demand for c in problem.customers)
        if max_demand > problem.vehicle_capacity:
            raise ValueError(
                f"Customer demand {max_demand} exceeds vehicle capacity {problem.vehicle_capacity}"
            )
        
        # Check if total demand can be served
        total_demand = sum(c.demand for c in problem.customers)
        max_total_capacity = problem.num_vehicles * problem.vehicle_capacity
        if total_demand > max_total_capacity:
            raise ValueError(
                f"Total demand {total_demand} exceeds total capacity {max_total_capacity}"
            )
        
        # Validate time windows
        for customer in problem.customers:
            if customer.ready_time > customer.due_date:
                raise ValueError(
                    f"Customer {customer.id} has invalid time window: "
                    f"ready_time ({customer.ready_time}) > due_date ({customer.due_date})"
                )
        
        # Validate distance matrix
        if problem.distance_matrix is not None:
            n = len(problem.customers) + 1  # +1 for depot
            if problem.distance_matrix.shape != (n, n):
                raise ValueError(
                    f"Distance matrix shape {problem.distance_matrix.shape} "
                    f"does not match expected {(n, n)}"
                )
        
        return True
    
    @staticmethod
    def validate_customer_data(customers: List[Dict]) -> bool:
        """
        Validate customer data dictionary.
        
        Args:
            customers: List of customer dictionaries
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If customer data is invalid
        """
        if not customers:
            raise ValueError("No customers provided")
        
        required_fields = ['id', 'x', 'y', 'demand', 'ready_time', 'due_date', 'service_time']
        
        for customer in customers:
            for field in required_fields:
                if field not in customer:
                    raise ValueError(f"Customer missing required field: {field}")
            
            if customer['demand'] < 0:
                raise ValueError(f"Customer {customer['id']} has negative demand")
            
            if customer['ready_time'] > customer['due_date']:
                raise ValueError(
                    f"Customer {customer['id']} has invalid time window"
                )
        
        return True
    
    @staticmethod
    def validate_depot_data(depot: Dict) -> bool:
        """
        Validate depot data dictionary.
        
        Args:
            depot: Depot dictionary
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If depot data is invalid
        """
        required_fields = ['id', 'x', 'y']
        
        for field in required_fields:
            if field not in depot:
                raise ValueError(f"Depot missing required field: {field}")
        
        return True
    
    @staticmethod
    def validate_chromosome(chromosome: List[int], problem: VRPProblem) -> bool:
        """
        Validate chromosome for correctness.
        
        Args:
            chromosome: Chromosome (list of customer IDs)
            problem: VRP problem instance
            
        Returns:
            True if valid, raises ValueError otherwise
            
        Raises:
            ValueError: If chromosome is invalid
        """
        if not chromosome:
            raise ValueError("Chromosome is empty")
        
        customer_ids = {c.id for c in problem.customers}
        chromosome_set = set(chromosome)
        
        # Check all customers are present
        if chromosome_set != customer_ids:
            missing = customer_ids - chromosome_set
            extra = chromosome_set - customer_ids
            raise ValueError(
                f"Chromosome invalid: missing {missing}, extra {extra}"
            )
        
        # Check no duplicates
        if len(chromosome) != len(set(chromosome)):
            raise ValueError("Chromosome contains duplicates")
        
        return True

