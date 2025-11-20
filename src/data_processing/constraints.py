"""
Constraint handling for VRP problems.
Validates capacity, time window, and other constraints.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import os
import time
import json
from src.models.vrp_model import VRPProblem
from src.core.pipeline_profiler import pipeline_profiler


class ConstraintHandler:
    """Handles VRP constraints validation and repair."""
    
    def __init__(self, vehicle_capacity: float, num_vehicles: int, penalty_weight: Optional[float] = None):
        """
        Initialize constraint handler.
        
        Args:
            vehicle_capacity: Maximum capacity per vehicle
            num_vehicles: Maximum number of vehicles available
            penalty_weight: Penalty weight for constraint violations (defaults to VRP_CONFIG)
        """
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles
        if penalty_weight is None:
            from config import VRP_CONFIG
            self.penalty_weight = VRP_CONFIG.get('penalty_weight', 5000)
        else:
            self.penalty_weight = penalty_weight
    
    def validate_capacity_constraint(self, 
                                  routes: List[List[int]], 
                                  demands: List[float]) -> Tuple[bool, float]:
        """
        Validate capacity constraints for all routes.
        
        Args:
            routes: List of routes, each route is a list of customer indices
            demands: List of customer demands (index 0 = customer ID 1, etc.)
            
        Returns:
            Tuple of (is_valid, total_penalty)
        """
        with pipeline_profiler.profile("constraints.capacity", metadata={'num_routes': len(routes)}):
            total_penalty = 0.0
            is_valid = True
            
            for route in routes:
                if not route:  # Empty route
                    continue
                
                route_load = 0.0
                for customer_id in route:
                    if customer_id != 0:  # Skip depot
                        # Customer ID 1 corresponds to demands[0], ID 2 to demands[1], etc.
                        if 1 <= customer_id <= len(demands):
                            route_load += demands[customer_id - 1]
                
                if route_load > self.vehicle_capacity:
                    is_valid = False
                    excess = route_load - self.vehicle_capacity
                    # Cap capacity penalty per route to prevent explosion
                    max_capacity_penalty = self.penalty_weight * 20  # Max 20 units excess per route
                    capacity_penalty = min(self.penalty_weight * excess, max_capacity_penalty)
                    total_penalty += capacity_penalty
            
            return is_valid, total_penalty
    
    def validate_vehicle_count_constraint(self, routes: List[List[int]]) -> Tuple[bool, float]:
        """
        Validate vehicle count constraint.
        
        Args:
            routes: List of routes
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        with pipeline_profiler.profile("constraints.vehicle_count", metadata={'num_routes': len(routes)}):
            num_used_vehicles = len([route for route in routes if route])
            
            if num_used_vehicles > self.num_vehicles:
                excess = num_used_vehicles - self.num_vehicles
                # Cap vehicle count penalty to max 5 extra vehicles
                max_vehicle_penalty = self.penalty_weight * 5
                penalty = min(self.penalty_weight * excess, max_vehicle_penalty)
                return False, penalty
            
            return True, 0.0
    
    def validate_customer_visit_constraint(self, 
                                        routes: List[List[int]], 
                                        customers: List) -> Tuple[bool, float]:
        """
        Validate that each customer is visited exactly once.
        
        Args:
            routes: List of routes
            customers: List of customer objects to get actual customer IDs
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        with pipeline_profiler.profile("constraints.customer_visit", metadata={'num_routes': len(routes)}):
            visited_customers = set()
            
            for route in routes:
                for customer_id in route:
                    if customer_id == 0:
                        continue  # ignore depot in visit constraint
                    if customer_id in visited_customers:
                        # Customer visited multiple times
                        return False, self.penalty_weight * 100
                    visited_customers.add(customer_id)
            
            # Check if all customers are visited - use actual customer IDs
            expected_customers = set(c.id for c in customers)
            missing_customers = expected_customers - visited_customers
            
            if missing_customers:
                # Cap missing customer penalty to max 10 missing customers
                max_missing_penalty = self.penalty_weight * 10
                penalty = min(self.penalty_weight * len(missing_customers), max_missing_penalty)
                return False, penalty
            
            return True, 0.0
    
    def validate_depot_constraint(self, routes: List[List[int]]) -> Tuple[bool, float]:
        """
        Validate that routes start and end at depot (customer 0).
        
        Args:
            routes: List of routes
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        with pipeline_profiler.profile("constraints.depot", metadata={'num_routes': len(routes)}):
            penalty = 0.0
            
            for route in routes:
                if not route:  # Empty route
                    continue
                
                # Check if route starts and ends at depot
                # Use smaller penalty for depot violations (less critical)
                depot_violation_penalty = self.penalty_weight * 0.5
                if route[0] != 0:
                    penalty += depot_violation_penalty
                if route[-1] != 0:
                    penalty += depot_violation_penalty
            
            is_valid = penalty == 0.0
            return is_valid, penalty
    
    def validate_time_window_constraint(self, 
                                      routes: List[List[int]], 
                                      time_windows: List[Tuple[float, float]],
                                      service_times: List[float],
                                      distance_matrix: np.ndarray,
                                      customers: Optional[List] = None,
                                      id_to_index: Optional[Dict[int, int]] = None) -> Tuple[bool, float]:
        """
        Validate time window constraints.
        
        Args:
            routes: List of routes (customer IDs)
            time_windows: List of (ready_time, due_date) for each customer (by customer order, not ID)
            service_times: List of service times for each customer (by customer order, not ID)
            distance_matrix: Distance matrix between all points (by matrix index, not customer ID)
            customers: Optional list of customer objects to map IDs to indices
            
        Returns:
            Tuple of (is_valid, penalty)
        """
        with pipeline_profiler.profile("constraints.time_windows", metadata={'num_routes': len(routes)}):
            total_penalty = 0.0
            
            # Get time window start from config (default 8:00 = 480 minutes)
            from config import VRP_CONFIG
            time_window_start = VRP_CONFIG.get('time_window_start', 480)
            
            # Create customer ID to index mapping if customers provided
            customer_id_to_index = {}
            if customers:
                for idx, customer in enumerate(customers):
                    customer_id_to_index[customer.id] = idx
            
            for route in routes:
                if len(route) < 2:  # Skip empty or single-point routes
                    continue
                
                current_time = float(time_window_start)  # Start from time window start, not 0
                
                for i in range(len(route) - 1):
                    from_customer_id = route[i]
                    to_customer_id = route[i + 1]
                    
                    # Skip depot (ID = 0)
                    if to_customer_id == 0:
                        continue
                    
                    # Get matrix indices for distance calculation
                    # Distance matrix uses matrix indices (0=depot, 1=first customer, etc.)
                    # We need to map customer IDs to matrix indices using id_to_index mapping
                    if id_to_index:
                        from_matrix_idx = id_to_index.get(from_customer_id, from_customer_id)
                        to_matrix_idx = id_to_index.get(to_customer_id, to_customer_id)
                    else:
                        # Fallback: assume customer ID = matrix index (may be incorrect)
                        from_matrix_idx = from_customer_id
                        to_matrix_idx = to_customer_id
                    
                    # If we have customer mapping, use it
                    if customers and to_customer_id in customer_id_to_index:
                        # Get customer object to access ready_time, due_date, service_time directly
                        customer = next((c for c in customers if c.id == to_customer_id), None)
                        if customer:
                            # Use customer object directly (more reliable)
                            ready_time = customer.ready_time
                            due_date = customer.due_date
                            service_time = customer.service_time
                        else:
                            # Fallback: use time_windows list with index mapping
                            customer_idx = customer_id_to_index[to_customer_id]
                            if 0 <= customer_idx < len(time_windows):
                                ready_time = time_windows[customer_idx][0]
                                due_date = time_windows[customer_idx][1]
                                service_time = service_times[customer_idx] if customer_idx < len(service_times) else 0.0
                            else:
                                continue  # Skip invalid customer
                    else:
                        # Fallback: assume customer ID = index in time_windows (may be incorrect)
                        if 0 <= to_customer_id - 1 < len(time_windows):
                            ready_time = time_windows[to_customer_id - 1][0]
                            due_date = time_windows[to_customer_id - 1][1]
                            service_time = service_times[to_customer_id - 1] if to_customer_id - 1 < len(service_times) else 0.0
                        else:
                            continue  # Skip invalid customer
                    
                    # Travel time from distance matrix (using matrix indices)
                    try:
                        travel_time = distance_matrix[from_matrix_idx, to_matrix_idx]
                    except (IndexError, KeyError):
                        # If matrix indexing fails, skip this segment
                        continue
                    
                    current_time += travel_time
                    
                    # Check if we arrive too early
                    if current_time < ready_time:
                        current_time = ready_time
                    
                    # Check if we arrive too late
                    if current_time > due_date:
                        # Tiered penalty system: graduated response based on severity
                        # Small violations get gentle penalties, large violations get strong penalties
                        lateness = current_time - due_date
                        
                        # Tier 1: Minor violations (< 10 time units)
                        if lateness < 10:
                            lateness_penalty = 100 * lateness
                        # Tier 2: Medium violations (10-60 time units)
                        elif lateness < 60:
                            lateness_penalty = 1000 + 500 * (lateness - 10)
                        # Tier 3: Major violations (>= 60 time units)
                        else:
                            lateness_penalty = 26000 + 1000 * (lateness - 60)
                        
                        total_penalty += lateness_penalty
                    
                    # Add service time
                    current_time += service_time
            
            is_valid = total_penalty == 0.0
            return is_valid, total_penalty
    
    def validate_all_constraints(self, 
                               routes: List[List[int]], 
                               demands: List[float],
                               customers: List,
                               time_windows: Optional[List[Tuple[float, float]]] = None,
                               service_times: Optional[List[float]] = None,
                               distance_matrix: Optional[np.ndarray] = None,
                               id_to_index: Optional[Dict[int, int]] = None) -> Dict:
        """
        Validate all VRP constraints.
        
        Args:
            routes: List of routes
            demands: List of customer demands
            customers: List of customer objects
            time_windows: Optional time windows for customers
            service_times: Optional service times for customers
            distance_matrix: Optional distance matrix
            
        Returns:
            Dictionary with validation results
        """
        with pipeline_profiler.profile("constraints.validate_all", metadata={'num_routes': len(routes)}):
            results = {
                'is_valid': True,
                'total_penalty': 0.0,
                'violations': {}
            }
            
            # Capacity constraint
            cap_valid, cap_penalty = self.validate_capacity_constraint(routes, demands)
            results['violations']['capacity'] = not cap_valid
            results['total_penalty'] += cap_penalty
            
            # Vehicle count constraint
            veh_valid, veh_penalty = self.validate_vehicle_count_constraint(routes)
            results['violations']['vehicle_count'] = not veh_valid
            results['total_penalty'] += veh_penalty
            
            # Customer visit constraint
            visit_valid, visit_penalty = self.validate_customer_visit_constraint(routes, customers)
            results['violations']['customer_visit'] = not visit_valid
            results['total_penalty'] += visit_penalty
            
            # Depot constraint
            depot_valid, depot_penalty = self.validate_depot_constraint(routes)
            results['violations']['depot'] = not depot_valid
            results['total_penalty'] += depot_penalty
            
            # Time window constraint (if data provided)
            if (time_windows is not None and service_times is not None and 
                distance_matrix is not None):
                tw_valid, tw_penalty = self.validate_time_window_constraint(
                    routes, time_windows, service_times, distance_matrix, 
                    customers=customers, id_to_index=id_to_index
                )
                results['violations']['time_windows'] = not tw_valid
                results['total_penalty'] += tw_penalty
            
            # Overall validity
            results['is_valid'] = not any(results['violations'].values())
            
            return results
    
    def repair_capacity_violations(self, 
                                 routes: List[List[int]], 
                                 demands: List[float]) -> List[List[int]]:
        """
        Repair capacity violations by moving customers to other routes.
        
        Args:
            routes: List of routes with potential capacity violations
            demands: List of customer demands
            
        Returns:
            Repaired routes
        """
        start_time = time.perf_counter()
        try:
            return self._repair_capacity_violations_impl(routes, demands)
        finally:
            pipeline_profiler.record(
                "constraints.repair_capacity",
                time.perf_counter() - start_time,
                metadata={'num_routes': len(routes)}
            )

    def _repair_capacity_violations_impl(self,
                                         routes: List[List[int]],
                                         demands: List[float]) -> List[List[int]]:
        """Internal implementation separated for profiling."""
        # Work on core customers only (strip depot 0)
        core_routes: List[List[int]] = []
        for route in routes:
            customers = [cid for cid in route if cid != 0]
            if customers:
                core_routes.append(customers)

        def safe_get_demand(cid: int) -> float:
            """Safely get demand for customer ID, handling out-of-bounds access."""
            if 1 <= cid <= len(demands):
                return demands[cid - 1]
            else:
                # If customer ID is beyond demands array, assume zero demand
                return 0.0

        def core_load(r: List[int]) -> float:
            return sum(safe_get_demand(cid) for cid in r)

        def remaining_capacity(r: List[int]) -> float:
            return self.vehicle_capacity - core_load(r)

        # 1) Build list of (idx, overflow) for overloaded routes
        def get_overloaded():
            over = []
            # Clean out any empty routes proactively
            non_empty = []
            for r in core_routes:
                if r:
                    non_empty.append(r)
            core_routes[:] = non_empty
            for idx, r in enumerate(core_routes):
                if not r:
                    continue
                load = core_load(r)
                if load > self.vehicle_capacity:
                    over.append((idx, load - self.vehicle_capacity))
            # Sort descending by overflow (fix the worst first)
            over.sort(key=lambda x: x[1], reverse=True)
            return over

        # Multi-pass Two-Phase Repair (increased from 5 to 10 for better repair)
        max_passes = 10
        ts = int(time.time())
        debug_log = []
        for pass_idx in range(max_passes):
            # Fast relocate: move lightest from most-overloaded to route with most space if fits
            changed = True
            relocate_iterations = 0
            max_relocate_iterations = 100  # Limit to prevent infinite loop
            while changed and relocate_iterations < max_relocate_iterations:
                changed = False
                relocate_iterations += 1
                # clean empties
                core_routes = [r for r in core_routes if r]
                if not core_routes:
                    break
                # find most overloaded route and target with most space
                overloads = sorted([(idx, core_load(r) - self.vehicle_capacity) for idx, r in enumerate(core_routes)], key=lambda x: x[1], reverse=True)
                if overloads and overloads[0][1] > 0:
                    oidx = overloads[0][0]
                    over_r = core_routes[oidx]
                    # lightest in overloaded
                    lp = min(range(len(over_r)), key=lambda j: safe_get_demand(over_r[j]))
                    need = safe_get_demand(over_r[lp])
                    # target with most space
                    targets = sorted([(j, self.vehicle_capacity - core_load(r)) for j, r in enumerate(core_routes) if j != oidx], key=lambda x: x[1], reverse=True)
                    if targets and targets[0][1] >= need:
                        tidx = targets[0][0]
                        core_routes[tidx].append(over_r.pop(lp))
                        changed = True
                    else:
                        break
                else:
                    break  # No more overloads

            # Phase 1: collect all overflow customers into a pool (while over routes safely)
            pool: List[int] = []
            i = 0
            max_route_iterations = 1000  # Safety limit
            route_iteration_count = 0
            while i < len(core_routes) and route_iteration_count < max_route_iterations:
                route_iteration_count += 1
                r = core_routes[i]
                if not r:
                    core_routes.pop(i)
                    continue
                # Limit iterations for removing customers from overloaded route
                remove_iterations = 0
                max_remove_iterations = 100
                while r and core_load(r) > self.vehicle_capacity and remove_iterations < max_remove_iterations:
                    remove_iterations += 1
                    # remove lightest customer
                    lightest_pos = min(range(len(r)), key=lambda j: safe_get_demand(r[j]))
                    pool.append(r.pop(lightest_pos))
                if not r:
                    core_routes.pop(i)
                    continue
                i += 1

            # If no overflow customers collected, we're done
            if not pool:
                break
            
            # Check vehicle count limit and merge routes if needed
            if len(core_routes) > self.num_vehicles:
                # Merge routes until within limit
                merge_iterations = 0
                max_merge_iterations = 50
                while len(core_routes) > self.num_vehicles and merge_iterations < max_merge_iterations:
                    merge_iterations += 1
                    if not core_routes or len(core_routes) < 2:
                        break
                    # Sort by load (ascending) to merge smallest routes first
                    core_routes.sort(key=lambda r: core_load(r))
                    # Merge two smallest routes
                    route1 = core_routes.pop(0)
                    route2 = core_routes.pop(0)
                    merged = route1 + route2
                    if core_load(merged) <= self.vehicle_capacity:
                        core_routes.append(merged)
                    else:
                        # Can't merge, restore routes
                        core_routes.insert(0, route2)
                        core_routes.insert(0, route1)
                        break

            # Phase 2: reinsert pool customers (ascending demand)
            pool.sort(key=lambda cid: safe_get_demand(cid))
            pool_total_demand = sum(safe_get_demand(cid) for cid in pool)
            total_space_before = sum(max(0.0, remaining_capacity(r)) for r in core_routes)
            # Optional light balancing before reinsertion: move lightest from near-full to most-space
            # Limit balancing iterations to prevent slowdown
            max_balancing_iterations = 10  # Reduced from 20
            for _bal in range(max_balancing_iterations):
                # find target with most remaining space
                if not core_routes:
                    break
                target = max(core_routes, key=lambda r: remaining_capacity(r))
                target_space = remaining_capacity(target)
                moved = False
                # consider donors sorted by load descending (near-full first)
                donors = sorted([r for r in core_routes if r is not target], key=lambda r: core_load(r), reverse=True)
                for donor in donors:
                    if not donor:
                        continue
                    # lightest in donor
                    lp = min(range(len(donor)), key=lambda j: safe_get_demand(donor[j]))
                    need = safe_get_demand(donor[lp])
                    if need <= target_space and (core_load(donor) - need) >= 0:
                        target.append(donor.pop(lp))
                        moved = True
                        break
                if not moved:
                    break
            for cid in pool:
                need = safe_get_demand(cid)
                # Recompute remaining capacity snapshot and sort routes by space desc
                routes_by_space = sorted(
                    core_routes,
                    key=lambda r: remaining_capacity(r),
                    reverse=True
                )
                placed = False
                for r in routes_by_space:
                    space = remaining_capacity(r)
                    if space >= need:
                        r.append(cid)
                        placed = True
                        break
                if not placed:
                    core_routes.append([cid])

            # Collect debug info for consolidated logging
            loads = [core_load(r) for r in core_routes]
            debug_info = {
                'pass': pass_idx + 1,
                'route_loads': loads,
                'pool_size_after_reinsert': 0,
                'pool_total_demand_before': pool_total_demand,
                'total_remaining_space_before': total_space_before,
                'num_routes': len(core_routes),
                'total_remaining_space': sum(max(0.0, self.vehicle_capacity - core_load(r)) for r in core_routes)
            }
            debug_log.append(debug_info)

        # Final check: if pool still has customers (shouldn't happen, but handle gracefully)
        # This can happen if total demand exceeds total capacity or reinsertion fails
        if pool:
            # Create new routes for remaining customers (may violate vehicle count, but at least all customers visited)
            pool_iterations = 0
            max_pool_iterations = 200  # Limit to prevent infinite loop
            while pool and pool_iterations < max_pool_iterations:
                pool_iterations += 1
                new_route = []
                # Try to fill route to capacity
                fill_iterations = 0
                max_fill_iterations = 100
                while pool and fill_iterations < max_fill_iterations:
                    fill_iterations += 1
                    customer = pool[0]
                    if core_load(new_route) + safe_get_demand(customer) <= self.vehicle_capacity:
                        new_route.append(pool.pop(0))
                    else:
                        break
                if new_route:
                    core_routes.append(new_route)
                else:
                    # Even single customer exceeds capacity - add anyway (violation, but better than missing customer)
                    if pool:
                        core_routes.append([pool.pop(0)])
                    else:
                        break
        
        # Final guarantee: if any route still exceeds capacity, split off lightest customers to new routes
        for i in range(len(core_routes)):
            split_iterations = 0
            max_split_iterations = 50  # Limit per route to prevent infinite loop
            while core_load(core_routes[i]) > self.vehicle_capacity and core_routes[i] and split_iterations < max_split_iterations:
                split_iterations += 1
                # remove lightest and create new route
                lp = min(range(len(core_routes[i])), key=lambda j: safe_get_demand(core_routes[i][j]))
                moved = core_routes[i].pop(lp)
                core_routes.append([moved])

        # 5) Wrap back with depot 0 at both ends
        repaired_routes: List[List[int]] = []
        for core in core_routes:
            if not core:
                continue
            repaired_routes.append([0] + core + [0])

        # Save consolidated debug log
        # Debug JSON files disabled to reduce clutter
        # Uncomment below if debugging is needed:
        # try:
        #     os.makedirs('results', exist_ok=True)
        #     consolidated_debug = {
        #         'timestamp': ts,
        #         'vehicle_capacity': self.vehicle_capacity,
        #         'repair_passes': debug_log,
        #         'final_routes': len(repaired_routes),
        #         'final_loads': [core_load(r) for r in core_routes]
        #     }
        #     with open(os.path.join('results', f'repair_debug_{ts}.json'), 'w', encoding='utf-8') as f:
        #         json.dump(consolidated_debug, f, indent=2, ensure_ascii=False)
        # except Exception:
        #     pass

        return repaired_routes
    
    def calculate_route_load(self, route: List[int], demands: List[float]) -> float:
        """Calculate total load for a route."""
        def safe_get_demand(cid: int) -> float:
            if 1 <= cid <= len(demands):
                return demands[cid - 1]
            else:
                return 0.0
        return sum(safe_get_demand(customer_id) for customer_id in route if customer_id != 0)
    
    def calculate_route_utilization(self, route: List[int], demands: List[float]) -> float:
        """Calculate utilization percentage for a route."""
        load = self.calculate_route_load(route, demands)
        return (load / self.vehicle_capacity) * 100 if self.vehicle_capacity > 0 else 0.0

    def _safe_get(self, arr, idx, default=0.0):
        try:
            return arr[idx]
        except Exception:
            return default

    # ---------------- Debug Utilities ----------------
    def analyze_routes(self, problem: VRPProblem, routes: List[List[int]]) -> Dict:
        """Produce a detailed analysis of routes and constraint status.
        Returns a dict with per-route loads, distances, start/end depot check,
        global coverage (missing/duplicate customers), and vehicle count usage.
        """
        analysis: Dict = {
            'vehicle_capacity': self.vehicle_capacity,
            'num_vehicles_allowed': self.num_vehicles,
            'num_routes_used': len([r for r in routes if r]),
            'routes': [],
            'coverage': {
                'expected_customers': len(problem.customers),
                'visited_customers': [],
                'missing_customers': [],
                'duplicate_customers': []
            },
            'summary': {}
        }

        demands = [c.demand for c in problem.customers]
        
        def safe_get_demand(cid: int) -> float:
            if 1 <= cid <= len(demands):
                return demands[cid - 1]
            else:
                return 0.0

        # Per-route analysis
        for idx, route in enumerate(routes):
            if not route:
                continue
            # load
            load = sum(safe_get_demand(cid) for cid in route if cid != 0)
            utilization = (load / self.vehicle_capacity) * 100 if self.vehicle_capacity > 0 else 0.0
            # distance
            distance = 0.0
            for i in range(len(route) - 1):
                distance += problem.get_distance(route[i], route[i + 1])
            # depot check
            starts_depot = (route[0] == 0)
            ends_depot = (route[-1] == 0)
            analysis['routes'].append({
                'route_index': idx,
                'route': route,
                'load': load,
                'utilization_percent': utilization,
                'distance': distance,
                'starts_depot': starts_depot,
                'ends_depot': ends_depot,
                'over_capacity': load > self.vehicle_capacity
            })

        # Coverage
        visited = []
        for route in routes:
            for cid in route:
                if cid != 0:
                    visited.append(cid)
        analysis['coverage']['visited_customers'] = sorted(visited)
        # duplicates
        seen = set()
        dups = []
        for cid in visited:
            if cid in seen:
                dups.append(cid)
            else:
                seen.add(cid)
        analysis['coverage']['duplicate_customers'] = sorted(list(set(dups)))
        expected = set(c.id for c in problem.customers)
        missing = sorted(list(expected - set(visited)))
        analysis['coverage']['missing_customers'] = missing

        # Summary
        analysis['summary'] = {
            'vehicle_overuse': analysis['num_routes_used'] > self.num_vehicles,
            'any_over_capacity': any(r['over_capacity'] for r in analysis['routes']),
            'any_missing_customers': len(missing) > 0,
            'any_duplicates': len(analysis['coverage']['duplicate_customers']) > 0,
            'any_bad_depot_route': any((not r['starts_depot'] or not r['ends_depot']) for r in analysis['routes'])
        }

        return analysis
