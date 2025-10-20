"""
Hanoi map visualization for VRP solutions.
Creates interactive maps using Folium for mockup datasets with real Hanoi coordinates.
"""

import folium
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
import json


class HanoiMapVisualizer:
    """Creates interactive Hanoi map visualizations for VRP solutions."""
    
    def __init__(self, problem: VRPProblem, config: Optional[Dict] = None):
        """
        Initialize Hanoi map visualizer.
        
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
    
    def create_map(self, individual: Individual, 
                   title: str = "VRP Routes - Hanoi",
                   save_path: Optional[str] = None) -> folium.Map:
        """
        Create interactive map with routes.
        
        Args:
            individual: VRP solution individual
            title: Map title
            save_path: Path to save map HTML file
            
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
            
            # Add route line
            route_coords = []
            for customer_id in route:
                if customer_id == 0:  # Depot
                    route_coords.append(depot_coords)
                else:
                    customer = self._get_customer_by_id(customer_id)
                    route_coords.append([customer.y, customer.x])
            
            # Draw route line
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Route {i+1}<br>Customers: {len(route)-1}<br>Load: {self._calculate_route_load(route)}"
            ).add_to(m)
            
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
        self._add_legend(m, routes)
        
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
            print(f"Map saved to: {save_path}")
        
        return m
    
    def create_comparison_map(self, ga_individual: Individual, 
                            nn_individual: Individual,
                            title: str = "VRP Comparison - Hanoi",
                            save_path: Optional[str] = None) -> folium.Map:
        """
        Create comparison map with GA and NN solutions.
        
        Args:
            ga_individual: GA solution
            nn_individual: NN solution
            title: Map title
            save_path: Path to save map HTML file
            
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
                weight=4,
                opacity=0.9,
                popup=f"GA Route {i+1}"
            ).add_to(m)
        
        # Add NN routes (dashed lines)
        nn_routes = self._decode_routes(nn_individual)
        for i, route in enumerate(nn_routes):
            if not route:
                continue
                
            color = self.colors[i % len(self.colors)]
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
        self._add_comparison_legend(m, ga_routes, nn_routes)
        
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
            print(f"Comparison map saved to: {save_path}")
        
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
    
    def _add_legend(self, m: folium.Map, routes: List[List[int]]):
        """Add route legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <p><b>Routes Legend</b></p>
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
    
    def _add_comparison_legend(self, m: folium.Map, ga_routes: List[List[int]], nn_routes: List[List[int]]):
        """Add comparison legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 250px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <p><b>Comparison Legend</b></p>
        <p><b>GA Solution:</b> Solid lines</p>
        <p><b>NN Solution:</b> Dashed lines</p>
        '''
        
        legend_html += '</div>'
        m.get_root().html.add_child(folium.Element(legend_html))


def create_hanoi_map(problem: VRPProblem, 
                    individual: Individual,
                    title: str = "VRP Routes - Hanoi",
                    save_path: Optional[str] = None) -> folium.Map:
    """
    Convenience function to create Hanoi map.
    
    Args:
        problem: VRP problem instance
        individual: VRP solution individual
        title: Map title
        save_path: Path to save map HTML file
        
    Returns:
        Folium map object
    """
    visualizer = HanoiMapVisualizer(problem)
    return visualizer.create_map(individual, title, save_path)


def create_hanoi_comparison_map(problem: VRPProblem,
                              ga_individual: Individual,
                              nn_individual: Individual,
                              title: str = "VRP Comparison - Hanoi",
                              save_path: Optional[str] = None) -> folium.Map:
    """
    Convenience function to create Hanoi comparison map.
    
    Args:
        problem: VRP problem instance
        ga_individual: GA solution
        nn_individual: NN solution
        title: Map title
        save_path: Path to save map HTML file
        
    Returns:
        Folium map object
    """
    visualizer = HanoiMapVisualizer(problem)
    return visualizer.create_comparison_map(ga_individual, nn_individual, title, save_path)
