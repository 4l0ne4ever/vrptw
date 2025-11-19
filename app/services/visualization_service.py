"""
Visualization service for VRP solutions.
Provides methods to create Folium maps and Plotly charts for Streamlit.
"""

import logging
from typing import Dict, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem

from app.config.settings import (
    OSRM_BASE_URL,
    OSRM_TIMEOUT,
    ROUTE_COLORS,
    USE_REAL_ROUTES,
)

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for creating visualizations of VRP solutions."""
    
    def __init__(self):
        """Initialize visualization service."""
        self.route_colors = ROUTE_COLORS
        self.use_real_routes = USE_REAL_ROUTES
        self.osrm_base_url = OSRM_BASE_URL.rstrip("/")
        self.osrm_timeout = OSRM_TIMEOUT
        self._routing_available = self.use_real_routes
        self._segment_cache: Dict[
            Tuple[float, float, float, float], Optional[List[Tuple[float, float]]]
        ] = {}
    
    def create_hanoi_map(
        self,
        solution: Individual,
        problem: VRPProblem,
        route_visibility: Optional[List[bool]] = None
    ) -> folium.Map:
        """
        Create Folium map for Hanoi dataset.
        
        Args:
            solution: VRP solution
            problem: VRP problem instance
            route_visibility: Optional list of booleans for route visibility
            
        Returns:
            Folium map object
        """
        if solution.is_empty() or not solution.routes:
            # Return empty map
            return folium.Map(location=[21.0285, 105.8542], zoom_start=12)
        
        # Calculate bounds
        all_points = [(problem.depot.y, problem.depot.x)]
        for route in solution.routes:
            for customer_id in route:
                if customer_id != 0:
                    customer = problem.get_customer_by_id(customer_id)
                    if customer:
                        all_points.append((customer.y, customer.x))
        
        if not all_points:
            return folium.Map(location=[21.0285, 105.8542], zoom_start=12)
        
        # Calculate center and bounds
        lats = [p[0] for p in all_points]
        lons = [p[1] for p in all_points]
        center_lat = (min(lats) + max(lats)) / 2
        center_lon = (min(lons) + max(lons)) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Fit bounds
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])
        
        # Add depot marker
        depot_coords = [problem.depot.y, problem.depot.x]
        num_routes = len([r for r in solution.routes if r])
        folium.Marker(
            depot_coords,
            popup=f"<b>Depot</b><br>Total Routes: {num_routes}",
            tooltip="Depot",
            icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
        ).add_to(m)
        
        # Add routes
        for i, route in enumerate(solution.routes):
            if not route or len(route) < 2:
                continue
            
            # Check visibility
            if route_visibility is not None and i < len(route_visibility):
                if not route_visibility[i]:
                    continue
            
            route_color = self.route_colors[i % len(self.route_colors)]
            
            # Get route coordinates (straight line for now - real routes would require routing API)
            route_coords = self._get_route_coordinates(route, problem)
            
            # Add polyline with unique identifier to avoid duplicate key errors
            route_polyline = folium.PolyLine(
                route_coords,
                color=route_color,
                weight=4,
                opacity=0.85,
                dash_array=None,
                popup=f"Route {i+1}",
                tooltip=f"Route {i+1}"
            )
            route_polyline.add_to(m)
            
            # Add customer markers with unique keys
            route_customer_idx = 0
            for j, customer_id in enumerate(route):
                if customer_id == 0:
                    continue
                
                customer = problem.get_customer_by_id(customer_id)
                if not customer:
                    continue
                
                # Create numbered marker with unique key
                marker_html = f"""
                <div style="
                    background-color: {route_color};
                    color: white;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    border: 2px solid white;
                ">{route_customer_idx}</div>
                """
                
                popup_html = f"""
                <b>Customer {customer_id}</b><br>
                Route: {i+1}<br>
                Position: {route_customer_idx}<br>
                Demand: {customer.demand}<br>
                Coordinates: ({customer.x:.4f}, {customer.y:.4f})
                """
                
                # Use unique key for each marker to avoid duplicate key errors
                customer_marker = folium.Marker(
                    [customer.y, customer.x],
                    popup=popup_html,
                    tooltip=f"C{customer_id} (R{i+1})",
                    icon=folium.DivIcon(
                        html=marker_html,
                        icon_size=(30, 30),
                        icon_anchor=(15, 15)
                    )
                )
                customer_marker.add_to(m)
                route_customer_idx += 1
        
        return m
    
    def create_solomon_chart(
        self,
        solution: Individual,
        problem: VRPProblem
    ) -> go.Figure:
        """
        Create Plotly scatter chart for Solomon dataset.
        
        Args:
            solution: VRP solution
            problem: VRP problem instance
            
        Returns:
            Plotly figure
        """
        if solution.is_empty() or not solution.routes:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(title="No Solution Available")
            return fig
        
        fig = go.Figure()
        
        # Add depot
        fig.add_trace(go.Scatter(
            x=[problem.depot.x],
            y=[problem.depot.y],
            mode='markers',
            name='Depot',
            marker=dict(
                size=15,
                color='red',
                symbol='square',
                line=dict(width=2, color='black')
            ),
            hovertemplate='<b>Depot</b><br>Coordinates: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ))
        
        # Add routes
        for i, route in enumerate(solution.routes):
            if not route or len(route) < 2:
                continue
            
            route_color = self.route_colors[i % len(self.route_colors)]
            
            # Extract customer coordinates
            customer_ids = [cid for cid in route if cid != 0]
            if not customer_ids:
                continue
            
            x_coords = []
            y_coords = []
            customer_labels = []
            
            for customer_id in customer_ids:
                customer = problem.get_customer_by_id(customer_id)
                if customer:
                    x_coords.append(customer.x)
                    y_coords.append(customer.y)
                    customer_labels.append(f"C{customer_id}")
            
            if not x_coords:
                continue
            
            # Add customer points
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                name=f'Route {i+1}',
                marker=dict(
                    size=10,
                    color=route_color,
                    line=dict(width=1, color='black')
                ),
                text=customer_labels,
                textposition='top center',
                hovertemplate='<b>%{text}</b><br>Route: ' + str(i+1) + '<br>Coordinates: (%{x:.2f}, %{y:.2f})<extra></extra>'
            ))
            
            # Add route lines (depot -> customers -> depot)
            route_x = [problem.depot.x]
            route_y = [problem.depot.y]
            for customer_id in customer_ids:
                customer = problem.get_customer_by_id(customer_id)
                if customer:
                    route_x.append(customer.x)
                    route_y.append(customer.y)
            route_x.append(problem.depot.x)
            route_y.append(problem.depot.y)
            
            fig.add_trace(go.Scatter(
                x=route_x,
                y=route_y,
                mode='lines',
                name=f'Route {i+1} Line',
                line=dict(color=route_color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title="VRP Solution - Route Visualization",
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            hovermode='closest',
            height=600
        )
        
        return fig
    
    def create_evolution_chart(
        self,
        evolution_data: List[Dict],
        show_avg: bool = True,
        log_scale: bool = False
    ) -> go.Figure:
        """
        Create evolution history chart.
        
        Args:
            evolution_data: List of generation data dictionaries
            show_avg: Whether to show average fitness
            log_scale: Whether to use log scale on y-axis
            
        Returns:
            Plotly figure
        """
        if not evolution_data:
            fig = go.Figure()
            fig.update_layout(title="No Evolution Data Available")
            return fig
        
        df = pd.DataFrame(evolution_data)
        
        fig = go.Figure()
        
        # Best fitness line
        if 'best_fitness' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['best_fitness'],
                mode='lines+markers',
                name='Best Fitness',
                line=dict(color='blue', width=2),
                marker=dict(size=5)
            ))
        
        # Best distance line
        if 'best_distance' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['best_distance'],
                mode='lines+markers',
                name='Best Distance',
                line=dict(color='green', width=2),
                marker=dict(size=5),
                yaxis='y2'
            ))
        
        # Average fitness line
        if show_avg and 'avg_fitness' in df.columns and df['avg_fitness'].notna().any():
            fig.add_trace(go.Scatter(
                x=df['generation'],
                y=df['avg_fitness'],
                mode='lines',
                name='Average Fitness',
                line=dict(color='orange', width=1, dash='dash')
            ))
        
        # Detect improvements (for markers)
        if 'best_fitness' in df.columns:
            improvements = []
            prev_fitness = None
            for idx, row in df.iterrows():
                if prev_fitness is not None and row['best_fitness'] > prev_fitness:
                    improvements.append(idx)
                prev_fitness = row['best_fitness']
            
            if improvements:
                improvement_gens = df.loc[improvements, 'generation']
                improvement_fitness = df.loc[improvements, 'best_fitness']
                fig.add_trace(go.Scatter(
                    x=improvement_gens,
                    y=improvement_fitness,
                    mode='markers',
                    name='Improvements',
                    marker=dict(size=10, color='red', symbol='star'),
                    showlegend=True
                ))
        
        yaxis_type = 'log' if log_scale else 'linear'
        
        fig.update_layout(
            title="Optimization Evolution",
            xaxis_title="Generation",
            yaxis_title="Fitness",
            yaxis=dict(type=yaxis_type, title="Fitness"),
            yaxis2=dict(
                title="Distance",
                overlaying='y',
                side='right',
                type=yaxis_type
            ) if 'best_distance' in df.columns else None,
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_comparison_chart(
        self,
        ga_solution: Individual,
        problem: VRPProblem,
        nn_solution: Optional[Individual] = None,
        bks_distance: Optional[float] = None
    ) -> go.Figure:
        """
        Create comparison bar chart.
        
        Args:
            ga_solution: GA solution
            problem: VRP problem instance
            nn_solution: Optional Nearest Neighbor solution
            bks_distance: Optional Best Known Solution distance
            
        Returns:
            Plotly figure
        """
        solutions = []
        distances = []
        colors = []
        labels = []
        
        # GA solution
        ga_distance = ga_solution.total_distance
        solutions.append("GA")
        distances.append(ga_distance)
        colors.append('#4ECDC4')
        labels.append(f"GA<br>{ga_distance:.2f}")
        
        # Nearest Neighbor solution
        if nn_solution and not nn_solution.is_empty():
            nn_distance = nn_solution.total_distance
            solutions.append("Nearest Neighbor")
            distances.append(nn_distance)
            
            # Color: green if better than GA, red if worse
            if nn_distance < ga_distance:
                colors.append('#90EE90')  # Light green
            else:
                colors.append('#FF6B6B')  # Light red
            
            improvement = ((nn_distance - ga_distance) / nn_distance) * 100
            labels.append(f"NN<br>{nn_distance:.2f}<br>({improvement:+.1f}%)")
        
        # BKS
        if bks_distance is not None:
            solutions.append("Best Known")
            distances.append(bks_distance)
            colors.append('#FFD700')  # Gold
            
            improvement = ((bks_distance - ga_distance) / bks_distance) * 100
            labels.append(f"BKS<br>{bks_distance:.2f}<br>({improvement:+.1f}%)")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=solutions,
            y=distances,
            marker_color=colors,
            text=labels,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Distance: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Solution Comparison",
            xaxis_title="Solution Method",
            yaxis_title="Total Distance",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def _get_route_coordinates(
        self,
        route: List[int],
        problem: VRPProblem
    ) -> List[Tuple[float, float]]:
        """
        Get coordinates for a route including depot.
        
        Args:
            route: Route as list of customer IDs
            problem: VRP problem instance
            
        Returns:
            List of (lat, lon) tuples
        """
        straight_coords = self._build_straight_route(route, problem)
        
        # Always use real routes for Hanoi (check coordinates to detect Hanoi)
        # Hanoi coordinates: lon ~105-106, lat ~20-21
        is_hanoi = False
        if straight_coords:
            first_coord = straight_coords[0]
            if 105.0 <= first_coord[1] <= 106.0 and 20.0 <= first_coord[0] <= 21.5:
                is_hanoi = True
        
        # Use real routes if enabled OR if it's Hanoi
        if self.use_real_routes or is_hanoi:
            road_coords = self._build_osrm_polyline(straight_coords)
            if road_coords:
                return road_coords
        
        return straight_coords

    def _build_straight_route(
        self,
        route: List[int],
        problem: VRPProblem
    ) -> List[Tuple[float, float]]:
        """Build straight-line coordinates depot -> customers -> depot."""
        coords: List[Tuple[float, float]] = [(problem.depot.y, problem.depot.x)]
        for customer_id in route:
            if customer_id == 0:
                continue
            customer = problem.get_customer_by_id(customer_id)
            if customer:
                coords.append((customer.y, customer.x))
        coords.append((problem.depot.y, problem.depot.x))
        return coords

    def _build_osrm_polyline(
        self,
        coords: List[Tuple[float, float]]
    ) -> Optional[List[Tuple[float, float]]]:
        """Build a road-aware polyline by chaining OSRM segments."""
        if not self._routing_available or len(coords) < 2:
            return coords
        
        road_coords: List[Tuple[float, float]] = [coords[0]]
        for start, end in zip(coords, coords[1:]):
            segment = self._fetch_osrm_segment(start, end)
            if segment and len(segment) >= 2:
                road_coords.extend(segment[1:])
            else:
                road_coords.append(end)
        
        # Remove duplicate consecutive points
        deduped: List[Tuple[float, float]] = [road_coords[0]]
        for lat, lon in road_coords[1:]:
            if lat != deduped[-1][0] or lon != deduped[-1][1]:
                deduped.append((lat, lon))
        
        return deduped

    def _fetch_osrm_segment(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """Fetch road geometry between two coordinates via OSRM."""
        key = (
            round(start[0], 4),
            round(start[1], 4),
            round(end[0], 4),
            round(end[1], 4),
        )
        if key in self._segment_cache:
            return self._segment_cache[key]
        
        start_lon = start[1]
        start_lat = start[0]
        end_lon = end[1]
        end_lat = end[0]
        coordinate_str = f"{start_lon:.6f},{start_lat:.6f};{end_lon:.6f},{end_lat:.6f}"
        url = f"{self.osrm_base_url}/route/v1/driving/{coordinate_str}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "false",
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.osrm_timeout)
            response.raise_for_status()
            data = response.json()
            routes = data.get("routes")
            if not routes:
                self._segment_cache[key] = None
                return None
            geometry = routes[0].get("geometry", {})
            coordinates = geometry.get("coordinates")
            if not coordinates:
                self._segment_cache[key] = None
                return None
            segment_coords = [(lat, lon) for lon, lat in coordinates]
            self._segment_cache[key] = segment_coords
            return segment_coords
        except requests.RequestException as exc:
            logger.warning(
                "OSRM request failed for segment %s -> %s: %s",
                start,
                end,
                exc,
            )
            self._routing_available = False
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "OSRM response parsing failed for segment %s -> %s: %s",
                start,
                end,
                exc,
            )
        
        self._segment_cache[key] = None
        return None

