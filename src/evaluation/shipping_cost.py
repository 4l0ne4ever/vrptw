"""
Shipping cost calculation module for VRP solutions.
Implements various shipping cost models including Ahamove pricing.
"""

from typing import Dict, List, Optional, Tuple
import math


class ShippingCostCalculator:
    """Calculates shipping costs for VRP solutions."""
    
    def __init__(self, cost_model: str = "ahamove"):
        """
        Initialize shipping cost calculator.
        
        Args:
            cost_model: Cost model to use ('ahamove', 'basic', 'custom')
        """
        self.cost_model = cost_model
        self._setup_cost_models()
    
    def _setup_cost_models(self):
        """Setup different cost models."""
        # Ahamove pricing model
        self.ahamove_pricing = {
            'express': {
                'base_price': 15709,  # VND for first 2km
                'price_2_3km': 19636,  # VND for 2-3km
                'price_per_km': 5400,   # VND per km after 3km
                'stop_fee': 5500,      # VND per additional stop
                'cod_fee_rate': 0.006,  # 0.6% COD fee
                'waiting_fee': 60000,   # VND per hour after free time
                'free_waiting_time': 15  # minutes
            },
            'standard': {
                'base_price': 12000,   # VND for first 2km
                'price_2_3km': 15000,  # VND for 2-3km
                'price_per_km': 4000,  # VND per km after 3km
                'stop_fee': 4000,      # VND per additional stop
                'cod_fee_rate': 0.005,  # 0.5% COD fee
                'waiting_fee': 50000,   # VND per hour after free time
                'free_waiting_time': 20  # minutes
            }
        }
        
        # Basic pricing model
        self.basic_pricing = {
            'price_per_km': 10000,     # VND per km
            'base_fee': 5000,         # VND base fee
            'stop_fee': 3000,         # VND per stop
            'cod_fee_rate': 0.004,     # 0.4% COD fee
            'waiting_fee': 40000,      # VND per hour
            'free_waiting_time': 10    # minutes
        }
    
    def calculate_route_cost(self, route: List[int], distances: List[float], 
                          demands: List[float], service_type: str = "express",
                          order_values: Optional[List[float]] = None,
                          waiting_times: Optional[List[float]] = None) -> Dict:
        """
        Calculate shipping cost for a single route.
        
        Args:
            route: List of customer IDs in route
            distances: List of distances between consecutive stops
            demands: List of demands for each customer
            service_type: Type of service ('express', 'standard')
            order_values: List of order values for COD calculation
            waiting_times: List of waiting times at each stop (minutes)
            
        Returns:
            Dictionary with cost breakdown
        """
        if self.cost_model == "ahamove":
            return self._calculate_ahamove_cost(route, distances, demands, 
                                              service_type, order_values, waiting_times)
        elif self.cost_model == "basic":
            return self._calculate_basic_cost(route, distances, demands, 
                                            order_values, waiting_times)
        else:
            raise ValueError(f"Unknown cost model: {self.cost_model}")
    
    def _calculate_ahamove_cost(self, route: List[int], distances: List[float],
                             demands: List[float], service_type: str,
                             order_values: Optional[List[float]] = None,
                             waiting_times: Optional[List[float]] = None) -> Dict:
        """Calculate cost using Ahamove pricing model."""
        pricing = self.ahamove_pricing[service_type]
        
        total_distance = sum(distances)
        total_cost = 0
        cost_breakdown = {
            'base_cost': 0,
            'distance_cost': 0,
            'stop_cost': 0,
            'cod_cost': 0,
            'waiting_cost': 0,
            'total_cost': 0
        }
        
        # Calculate base distance cost
        if total_distance <= 2:
            cost_breakdown['base_cost'] = pricing['base_price']
        elif total_distance <= 3:
            cost_breakdown['base_cost'] = pricing['base_price'] + pricing['price_2_3km']
        else:
            cost_breakdown['base_cost'] = (pricing['base_price'] + pricing['price_2_3km'] + 
                                         pricing['price_per_km'] * (total_distance - 3))
        
        cost_breakdown['distance_cost'] = cost_breakdown['base_cost']
        
        # Calculate stop cost (additional stops beyond depot)
        num_stops = len(route) - 1  # Exclude depot
        if num_stops > 1:
            cost_breakdown['stop_cost'] = pricing['stop_fee'] * (num_stops - 1)
        
        # Calculate COD cost
        if order_values:
            total_order_value = sum(order_values)
            cost_breakdown['cod_cost'] = total_order_value * pricing['cod_fee_rate']
        
        # Calculate waiting cost
        if waiting_times:
            total_waiting_time = sum(waiting_times)
            free_time_minutes = pricing['free_waiting_time']
            
            if total_waiting_time > free_time_minutes:
                excess_time_hours = (total_waiting_time - free_time_minutes) / 60
                cost_breakdown['waiting_cost'] = pricing['waiting_fee'] * excess_time_hours
        
        # Calculate total cost
        cost_breakdown['total_cost'] = (cost_breakdown['distance_cost'] + 
                                      cost_breakdown['stop_cost'] + 
                                      cost_breakdown['cod_cost'] + 
                                      cost_breakdown['waiting_cost'])
        
        return cost_breakdown
    
    def _calculate_basic_cost(self, route: List[int], distances: List[float],
                            demands: List[float], order_values: Optional[List[float]] = None,
                            waiting_times: Optional[List[float]] = None) -> Dict:
        """Calculate cost using basic pricing model."""
        pricing = self.basic_pricing
        
        total_distance = sum(distances)
        cost_breakdown = {
            'base_cost': pricing['base_fee'],
            'distance_cost': total_distance * pricing['price_per_km'],
            'stop_cost': 0,
            'cod_cost': 0,
            'waiting_cost': 0,
            'total_cost': 0
        }
        
        # Calculate stop cost
        num_stops = len(route) - 1  # Exclude depot
        if num_stops > 0:
            cost_breakdown['stop_cost'] = pricing['stop_fee'] * num_stops
        
        # Calculate COD cost
        if order_values:
            total_order_value = sum(order_values)
            cost_breakdown['cod_cost'] = total_order_value * pricing['cod_fee_rate']
        
        # Calculate waiting cost
        if waiting_times:
            total_waiting_time = sum(waiting_times)
            free_time_minutes = pricing['free_waiting_time']
            
            if total_waiting_time > free_time_minutes:
                excess_time_hours = (total_waiting_time - free_time_minutes) / 60
                cost_breakdown['waiting_cost'] = pricing['waiting_fee'] * excess_time_hours
        
        # Calculate total cost
        cost_breakdown['total_cost'] = (cost_breakdown['base_cost'] + 
                                      cost_breakdown['distance_cost'] + 
                                      cost_breakdown['stop_cost'] + 
                                      cost_breakdown['cod_cost'] + 
                                      cost_breakdown['waiting_cost'])
        
        return cost_breakdown
    
    def calculate_solution_cost(self, routes: List[List[int]], problem,
                             service_type: str = "express",
                             order_values: Optional[List[float]] = None,
                             waiting_times: Optional[List[float]] = None) -> Dict:
        """
        Calculate total shipping cost for entire solution.
        
        Args:
            routes: List of routes (each route is list of customer IDs)
            problem: VRP problem instance
            service_type: Type of service ('express', 'standard')
            order_values: List of order values for each customer
            waiting_times: List of waiting times for each customer
            
        Returns:
            Dictionary with total cost and breakdown by route
        """
        total_cost = 0
        route_costs = []
        
        for route_idx, route in enumerate(routes):
            if not route:
                continue
            
            # Calculate distances for this route
            distances = []
            for i in range(len(route) - 1):
                current_id = route[i]
                next_id = route[i + 1]
                distance = problem.get_distance(current_id, next_id)
                distances.append(distance)
            
            # Get demands for this route
            route_demands = []
            route_order_values = []
            route_waiting_times = []
            
            for customer_id in route[1:-1]:  # Exclude depot
                customer = problem.get_customer_by_id(customer_id)
                if customer:
                    route_demands.append(customer.demand)
                    
                    if order_values and customer_id <= len(order_values):
                        route_order_values.append(order_values[customer_id - 1])
                    
                    if waiting_times and customer_id <= len(waiting_times):
                        route_waiting_times.append(waiting_times[customer_id - 1])
            
            # Calculate cost for this route
            route_cost = self.calculate_route_cost(
                route, distances, route_demands, service_type,
                route_order_values if route_order_values else None,
                route_waiting_times if route_waiting_times else None
            )
            
            route_costs.append({
                'route_id': route_idx + 1,
                'route': route,
                'distance': sum(distances),
                'cost_breakdown': route_cost
            })
            
            total_cost += route_cost['total_cost']
        
        return {
            'total_cost': total_cost,
            'route_costs': route_costs,
            'service_type': service_type,
            'cost_model': self.cost_model
        }
    
    def generate_order_values(self, customers, value_range: Tuple[float, float] = (100000, 1000000)) -> List[float]:
        """
        Generate random order values for customers.
        
        Args:
            customers: List of customer objects
            value_range: Tuple of (min_value, max_value) in VND
            
        Returns:
            List of order values
        """
        import random
        
        order_values = []
        for customer in customers:
            # Generate order value based on demand (higher demand = higher value)
            base_value = random.uniform(value_range[0], value_range[1])
            # Adjust based on demand (assume 1 unit demand = 10,000 VND)
            adjusted_value = base_value + (customer.demand * 10000)
            order_values.append(adjusted_value)
        
        return order_values
    
    def generate_waiting_times(self, customers, time_range: Tuple[float, float] = (5, 30)) -> List[float]:
        """
        Generate random waiting times for customers.
        
        Args:
            customers: List of customer objects
            time_range: Tuple of (min_time, max_time) in minutes
            
        Returns:
            List of waiting times
        """
        import random
        
        waiting_times = []
        for customer in customers:
            # Generate waiting time based on service time
            base_time = random.uniform(time_range[0], time_range[1])
            # Add service time
            total_time = base_time + customer.service_time
            waiting_times.append(total_time)
        
        return waiting_times


def calculate_shipping_cost_example():
    """Example calculation following the provided formula."""
    # Example from the description
    distance_km = 5
    num_stops = 2  # 2 delivery points
    
    # Ahamove Express pricing
    base_price = 15709  # First 2km
    price_2_3km = 19636  # 2-3km
    price_per_km = 5400  # Per km after 3km
    stop_fee = 5500  # Per additional stop
    
    # Calculate distance cost
    if distance_km <= 2:
        distance_cost = base_price
    elif distance_km <= 3:
        distance_cost = base_price + price_2_3km
    else:
        distance_cost = base_price + price_2_3km + price_per_km * (distance_km - 3)
    
    # Calculate stop cost
    stop_cost = stop_fee * (num_stops - 1)  # Additional stops beyond first
    
    # Total cost
    total_cost = distance_cost + stop_cost
    
    print(f"Distance cost: {distance_cost:,} VND")
    print(f"Stop cost: {stop_cost:,} VND")
    print(f"Total cost: {total_cost:,} VND")
    
    return total_cost


if __name__ == "__main__":
    # Test the example calculation
    calculate_shipping_cost_example()
