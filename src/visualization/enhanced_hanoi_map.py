"""
Enhanced Hanoi map visualization with real routing.
Creates interactive maps using Folium with real road routes instead of straight lines.
"""

import folium
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from src.data_processing.enhanced_hanoi_coordinates import EnhancedHanoiCoordinateGenerator
import json


class EnhancedHanoiMapVisualizer:
    """Creates interactive Hanoi map visualizations with real routing."""
    
    def __init__(self, problem: VRPProblem, config: Optional[Dict] = None):
        """
        Initialize enhanced Hanoi map visualizer.
        
        Args:
            problem: VRP problem instance
            config: Visualization configuration
        """
        self.problem = problem
        self.config = config or {}
        
        # Hanoi center coordinates
        self.hanoi_center = [21.0285, 105.8542]  # Hoan Kiem Lake
        self.default_zoom = 12
        
        # Vehicle colors
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        # Initialize enhanced coordinate generator
        self.coord_generator = EnhancedHanoiCoordinateGenerator()
        
        # Cache for routes to avoid repeated API calls
        self.route_cache = {}
    
    def create_map(self, individual: Individual, 
                   title: str = "VRP Routes - Hanoi (Real Routes)",
                   save_path: Optional[str] = None,
                   use_real_routes: bool = True) -> folium.Map:
        """
        Create interactive map with real routes.
        
        Args:
            individual: VRP solution individual
            title: Map title
            save_path: Path to save map HTML file
            use_real_routes: Whether to use real routing or straight lines
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=self.hanoi_center,
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )
        
        # Decode routes
        routes = self._decode_routes(individual)
        
        # Add depot
        depot_coords = [self.problem.depot.y, self.problem.depot.x]  # [lat, lon]
        folium.Marker(
            depot_coords,
            popup=f"Depot<br>Capacity: {self.problem.vehicle_capacity}",
            tooltip="Depot",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)
        
        # Add routes
        for i, route in enumerate(routes):
            if not route:
                continue
                
            color = self.colors[i % len(self.colors)]
            
            if use_real_routes:
                # Use real routing
                self._add_real_route(m, route, color, i)
            else:
                # Use straight lines (fallback)
                self._add_straight_route(m, route, color, i)
            
            # Add customer markers
            for j, customer_id in enumerate(route):
                if customer_id == 0:  # Skip depot (already added)
                    continue
                    
                customer = self._get_customer_by_id(customer_id)
                customer_coords = [customer.y, customer.x]
                
                folium.CircleMarker(
                    customer_coords,
                    radius=8,
                    popup=f"Customer {customer_id}<br>Demand: {customer.demand}<br>Route: {i+1}",
                    tooltip=f"C{customer_id}",
                    color=color,
                    fill=True,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add legend
        self._add_legend(m, routes, use_real_routes)
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%);
                    z-index:9999; font-size:16px; font-weight: bold;
                    background-color: white; padding: 5px 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        {title}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Save map if path provided
        if save_path:
            m.save(save_path)
            print(f"Enhanced map saved to: {save_path}")
        
        return m
    
    def _add_real_route(self, m: folium.Map, route: List[int], color: str, route_id: int):
        """Add real route to map."""
        route_coords = []
        for customer_id in route:
            if customer_id == 0:  # Depot
                depot_coords = [self.problem.depot.y, self.problem.depot.x]
                route_coords.append(depot_coords)
            else:
                customer = self._get_customer_by_id(customer_id)
                route_coords.append([customer.y, customer.x])
        
        # Draw real route segments
        for i in range(len(route_coords) - 1):
            start = route_coords[i]
            end = route_coords[i + 1]
            
            # Get real route between two points
            real_route = self._get_cached_route(start, end)
            
            if real_route and len(real_route) > 1:
                # Draw real route
                folium.PolyLine(
                    real_route,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Route {route_id+1} Segment {i+1}"
                ).add_to(m)
            else:
                # Fallback to straight line
                folium.PolyLine(
                    [start, end],
                    color=color,
                    weight=2,
                    opacity=0.6,
                    dash_array='5, 5',
                    popup=f"Route {route_id+1} Segment {i+1} (Straight)"
                ).add_to(m)
    
    def _add_straight_route(self, m: folium.Map, route: List[int], color: str, route_id: int):
        """Add straight line route to map."""
        route_coords = []
        for customer_id in route:
            if customer_id == 0:  # Depot
                depot_coords = [self.problem.depot.y, self.problem.depot.x]
                route_coords.append(depot_coords)
            else:
                customer = self._get_customer_by_id(customer_id)
                route_coords.append([customer.y, customer.x])
        
        # Draw straight line route
        folium.PolyLine(
            route_coords,
            color=color,
            weight=3,
            opacity=0.8,
            popup=f"Route {route_id+1}<br>Customers: {len(route)-1}<br>Load: {self._calculate_route_load(route)}"
        ).add_to(m)
    
    def _get_cached_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Get route from cache or API."""
        # Create cache key
        cache_key = f"{start[0]:.6f},{start[1]:.6f}_{end[0]:.6f},{end[1]:.6f}"
        
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        # Get route from API
        route = self.coord_generator.get_real_route(start, end)
        
        # Cache the result
        self.route_cache[cache_key] = route
        
        return route
    
    def create_comparison_map(self, ga_individual: Individual, 
                            nn_individual: Individual,
                            title: str = "VRP Comparison - Hanoi (Real Routes)",
                            save_path: Optional[str] = None,
                            use_real_routes: bool = True) -> folium.Map:
        """
        Create comparison map with GA and NN solutions using real routes.
        
        Args:
            ga_individual: GA solution
            nn_individual: NN solution
            title: Map title
            save_path: Path to save map HTML file
            use_real_routes: Whether to use real routing
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=self.hanoi_center,
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )
        
        # Add depot
        depot_coords = [self.problem.depot.y, self.problem.depot.x]
        folium.Marker(
            depot_coords,
            popup="Depot",
            tooltip="Depot",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)
        
        # Add GA routes (solid lines)
        ga_routes = self._decode_routes(ga_individual)
        for i, route in enumerate(ga_routes):
            if not route:
                continue
                
            color = self.colors[i % len(self.colors)]
            
            if use_real_routes:
                self._add_real_route(m, route, color, i)
            else:
                self._add_straight_route(m, route, color, i)
        
        # Add NN routes (dashed lines)
        nn_routes = self._decode_routes(nn_individual)
        for i, route in enumerate(nn_routes):
            if not route:
                continue
                
            color = self.colors[i % len(self.colors)]
            
            # NN routes with dashed style
            route_coords = []
            for customer_id in route:
                if customer_id == 0:
                    route_coords.append(depot_coords)
                else:
                    customer = self._get_customer_by_id(customer_id)
                    route_coords.append([customer.y, customer.x])
            
            folium.PolyLine(
                route_coords,
                color=color,
                weight=2,
                opacity=0.6,
                dash_array='10, 10',
                popup=f"NN Route {i+1}"
            ).add_to(m)
        
        # Add customer markers
        for customer in self.problem.customers:
            customer_coords = [customer.y, customer.x]
            folium.CircleMarker(
                customer_coords,
                radius=6,
                popup=f"Customer {customer.id}<br>Demand: {customer.demand}",
                tooltip=f"C{customer.id}",
                color='black',
                fill=True,
                fillOpacity=0.8
            ).add_to(m)
        
        # Add comparison legend
        self._add_comparison_legend(m, ga_routes, nn_routes, use_real_routes)
        
        # Add title
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%);
                    z-index:9999; font-size:16px; font-weight: bold;
                    background-color: white; padding: 5px 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        {title}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        if save_path:
            m.save(save_path)
            print(f"Enhanced comparison map saved to: {save_path}")
        
        return m
    
    def _decode_routes(self, individual: Individual) -> List[List[int]]:
        """Decode individual chromosome into routes."""
        from src.algorithms.decoder import RouteDecoder
        decoder = RouteDecoder(self.problem)
        return decoder.decode_chromosome(individual.chromosome)
    
    def _get_customer_by_id(self, customer_id: int):
        """Get customer object by ID."""
        for customer in self.problem.customers:
            if customer.id == customer_id:
                return customer
        raise ValueError(f"Customer {customer_id} not found")
    
    def _calculate_route_load(self, route: List[int]) -> float:
        """Calculate total load of a route."""
        total_load = 0.0
        for customer_id in route:
            if customer_id != 0:
                customer = self._get_customer_by_id(customer_id)
                total_load += customer.demand
        return total_load
    
    def _add_legend(self, m: folium.Map, routes: List[List[int]], use_real_routes: bool):
        """Add route legend to map."""
        route_type = "Real Routes" if use_real_routes else "Straight Lines"
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 250px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <p><b>Routes Legend ({route_type})</b></p>
        '''
        
        for i, route in enumerate(routes):
            if route:
                color = self.colors[i % len(self.colors)]
                load = self._calculate_route_load(route)
                legend_html += f'''
                <p><i class="fa fa-circle" style="color:{color}"></i> 
                   Route {i+1}: {len(route)-1} customers, Load: {load:.1f}</p>
                '''
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_comparison_legend(self, m: folium.Map, ga_routes: List[List[int]], nn_routes: List[List[int]], use_real_routes: bool):
        """Add comparison legend to map."""
        route_type = "Real Routes" if use_real_routes else "Straight Lines"
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 280px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <p><b>Comparison Legend ({route_type})</b></p>
        <p><b>GA Solution:</b> Solid lines</p>
        <p><b>NN Solution:</b> Dashed lines</p>
        '''
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))


def create_enhanced_hanoi_map(problem: VRPProblem, 
                            individual: Individual,
                            title: str = "VRP Routes - Hanoi (Real Routes)",
                            save_path: Optional[str] = None,
                            use_real_routes: bool = True) -> folium.Map:
    """
    Convenience function to create enhanced Hanoi map.
    
    Args:
        problem: VRP problem instance
        individual: VRP solution individual
        title: Map title
        save_path: Path to save map HTML file
        use_real_routes: Whether to use real routing
        
    Returns:
        Folium map object
    """
    visualizer = EnhancedHanoiMapVisualizer(problem)
    return visualizer.create_map(individual, title, save_path, use_real_routes)


def create_enhanced_hanoi_comparison_map(problem: VRPProblem,
                                       ga_individual: Individual,
                                       nn_individual: Individual,
                                       title: str = "VRP Comparison - Hanoi (Real Routes)",
                                       save_path: Optional[str] = None,
                                       use_real_routes: bool = True) -> folium.Map:
    """
    Convenience function to create enhanced Hanoi comparison map.
    
    Args:
        problem: VRP problem instance
        ga_individual: GA solution
        nn_individual: NN solution
        title: Map title
        save_path: Path to save map HTML file
        use_real_routes: Whether to use real routing
        
    Returns:
        Folium map object
    """
    visualizer = EnhancedHanoiMapVisualizer(problem)
    return visualizer.create_comparison_map(ga_individual, nn_individual, title, save_path, use_real_routes)
