"""
Route mapping and visualization for VRP solutions.
Creates 2D plots of routes with different colors per vehicle.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Optional, Tuple
import numpy as np
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from config import VIZ_CONFIG


class RouteMapper:
    """Creates route visualizations for VRP solutions."""
    
    def __init__(self, problem: VRPProblem, config: Optional[Dict] = None):
        """
        Initialize route mapper.
        
        Args:
            problem: VRP problem instance
            config: Visualization configuration
        """
        self.problem = problem
        self.config = config or VIZ_CONFIG.copy()
        
        # Set up matplotlib
        plt.style.use('default')
        self.fig_size = self.config['figure_size']
        self.dpi = self.config['dpi']
        self.colors = self.config['colors']
        self.marker_size = self.config['marker_size']
        self.line_width = self.config['line_width']
        self.font_size = self.config['font_size']
    
    def plot_routes(self, individual: Individual, 
                   title: str = "VRP Solution",
                   save_path: Optional[str] = None,
                   show_labels: bool = True) -> plt.Figure:
        """
        Plot routes for a VRP solution.
        
        Args:
            individual: Solution individual
            title: Plot title
            save_path: Optional path to save plot
            show_labels: Whether to show customer labels
            
        Returns:
            Matplotlib figure
        """
        if individual.is_empty():
            raise ValueError("Cannot plot empty solution")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot depot
        depot_x, depot_y = self.problem.depot.x, self.problem.depot.y
        ax.scatter(depot_x, depot_y, c='#000000', s=self.marker_size * 2, 
                  marker='s', label='Depot', zorder=5)
        
        # Plot customers
        customer_coords = self.problem.get_customer_coordinates()
        customer_x = [coord[0] for coord in customer_coords]
        customer_y = [coord[1] for coord in customer_coords]
        
        ax.scatter(customer_x, customer_y, c='#D3D3D3', s=self.marker_size, 
                  marker='o', label='Customers', zorder=3)
        
        # Plot routes
        routes = individual.routes
        if not routes:
            routes = self._decode_routes(individual)
        
        for i, route in enumerate(routes):
            if not route or len(route) < 2:
                continue
            
            color = self.colors[i % len(self.colors)]
            self._plot_single_route(ax, route, color, i, show_labels)
        
        # Customize plot
        ax.set_xlabel('X Coordinate', fontsize=self.font_size)
        ax.set_ylabel('Y Coordinate', fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')
        ax.legend(fontsize=self.font_size - 2)
        ax.grid(True)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics text
        self._add_statistics_text(ax, individual)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _plot_single_route(self, ax, route: List[int], color: str, 
                          route_num: int, show_labels: bool):
        """Plot a single route."""
        if len(route) < 2:
            return
        
        # Plot route line
        route_x = []
        route_y = []
        
        for customer_id in route:
            if customer_id == 0:  # Depot
                route_x.append(self.problem.depot.x)
                route_y.append(self.problem.depot.y)
            else:
                customer = self.problem.get_customer_by_id(customer_id)
                if customer is not None:
                    route_x.append(customer.x)
                    route_y.append(customer.y)
                else:
                    print(f"Warning: Customer {customer_id} not found, skipping...")
        
        # Draw route line
        ax.plot(route_x, route_y, color=color, linewidth=self.line_width, 
               alpha=0.7, label=f'Route {route_num + 1}', zorder=2)
        
        # Draw arrows to show direction
        for i in range(len(route_x) - 1):
            dx = route_x[i + 1] - route_x[i]
            dy = route_y[i + 1] - route_y[i]
            
            # Normalize for arrow length
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx = dx / length * 0.5
                dy = dy / length * 0.5
                
                ax.arrow(route_x[i], route_y[i], dx, dy, 
                        head_width=0.3, head_length=0.2, 
                        fc=color, ec=color, alpha=0.8, zorder=4)
        
        # Highlight route customers
        customer_x = [route_x[i] for i in range(1, len(route_x) - 1)]  # Exclude depot
        customer_y = [route_y[i] for i in range(1, len(route_y) - 1)]  # Exclude depot
        
        ax.scatter(customer_x, customer_y, c=color, s=self.marker_size, 
                  marker='o', alpha=0.8, zorder=4)
        
        # Add customer labels
        if show_labels:
            for i, customer_id in enumerate(route[1:-1]):  # Exclude depot visits
                ax.annotate(str(customer_id), 
                           (customer_x[i], customer_y[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=self.font_size - 4, color=color,
                           fontweight='bold')
    
    def _decode_routes(self, individual: Individual) -> List[List[int]]:
        """Decode routes from chromosome if not available."""
        from src.algorithms.decoder import RouteDecoder
        decoder = RouteDecoder(self.problem)
        return decoder.decode_chromosome(individual.chromosome)
    
    def _add_statistics_text(self, ax, individual: Individual):
        """Add statistics text to the plot."""
        stats_text = f"Total Distance: {individual.total_distance:.2f}\n"
        stats_text += f"Routes: {individual.get_route_count()}\n"
        stats_text += f"Customers: {individual.get_customer_count()}\n"
        stats_text += f"Fitness: {individual.fitness:.4f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=self.font_size - 2, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat'))
    
    def plot_comparison(self, solutions: List[Individual], 
                       solution_names: List[str],
                       title: str = "VRP Solutions Comparison",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple solutions.
        
        Args:
            solutions: List of solutions to compare
            solution_names: Names for solutions
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not solutions:
            raise ValueError("No solutions to plot")
        
        n_solutions = len(solutions)
        fig, axes = plt.subplots(1, n_solutions, figsize=(self.fig_size[0] * n_solutions, self.fig_size[1]))
        
        if n_solutions == 1:
            axes = [axes]
        
        for i, (solution, name) in enumerate(zip(solutions, solution_names)):
            ax = axes[i]
            
            # Plot depot
            depot_x, depot_y = self.problem.depot.x, self.problem.depot.y
            ax.scatter(depot_x, depot_y, c='black', s=self.marker_size * 2, 
                      marker='s', label='Depot', zorder=5)
            
            # Plot customers
            customer_coords = self.problem.get_customer_coordinates()
            customer_x = [coord[0] for coord in customer_coords]
            customer_y = [coord[1] for coord in customer_coords]
            
            ax.scatter(customer_x, customer_y, c='lightgray', s=self.marker_size, 
                      marker='o', label='Customers', zorder=3)
            
            # Plot routes
            routes = solution.routes
            if not routes:
                routes = self._decode_routes(solution)
            
            for j, route in enumerate(routes):
                if not route or len(route) < 2:
                    continue
                
                color = self.colors[j % len(self.colors)]
                self._plot_single_route(ax, route, color, j, False)
            
            # Customize subplot
            ax.set_title(f"{name}\nDistance: {solution.total_distance:.2f}", 
                        fontsize=self.font_size)
            ax.set_xlabel('X Coordinate', fontsize=self.font_size - 2)
            ax.set_ylabel('Y Coordinate', fontsize=self.font_size - 2)
            ax.grid(True)
            ax.set_aspect('equal', adjustable='box')
        
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_route_loads(self, individual: Individual,
                        title: str = "Route Load Distribution",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot route load distribution.
        
        Args:
            individual: Solution individual
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        if individual.is_empty():
            raise ValueError("Cannot plot empty solution")
        
        routes = individual.routes
        if not routes:
            routes = self._decode_routes(individual)
        
        # Calculate route loads
        route_loads = []
        route_ids = []
        
        for i, route in enumerate(routes):
            if route:
                route_load = sum(
                    self.problem.get_customer_by_id(c).demand 
                    for c in route if c != 0
                )
                route_loads.append(route_load)
                route_ids.append(f"Route {i + 1}")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_size[0] * 1.5, self.fig_size[1]))
        
        # Bar chart
        colors = [self.colors[i % len(self.colors)] for i in range(len(route_loads))]
        bars = ax1.bar(route_ids, route_loads, color=colors, alpha=0.7)
        
        # Add capacity line
        ax1.axhline(y=self.problem.vehicle_capacity, color='red', 
                   linestyle='--', linewidth=2, label='Capacity')
        
        ax1.set_xlabel('Route', fontsize=self.font_size)
        ax1.set_ylabel('Load', fontsize=self.font_size)
        ax1.set_title('Route Loads', fontsize=self.font_size)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, load in zip(bars, route_loads):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{load:.1f}', ha='center', va='bottom', fontsize=self.font_size - 2)
        
        # Pie chart for load distribution
        if route_loads:
            ax2.pie(route_loads, labels=route_ids, colors=colors, autopct='%1.1f%%',
                   startangle=90)
            ax2.set_title('Load Distribution', fontsize=self.font_size)
        
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_customer_demands(self, title: str = "Customer Demands",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot customer demand distribution.
        
        Args:
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Get customer coordinates and demands
        customer_coords = self.problem.get_customer_coordinates()
        demands = self.problem.get_demands()
        
        customer_x = [coord[0] for coord in customer_coords]
        customer_y = [coord[1] for coord in customer_coords]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size, dpi=self.dpi)
        
        # Plot depot
        depot_x, depot_y = self.problem.depot.x, self.problem.depot.y
        ax.scatter(depot_x, depot_y, c='#000000', s=self.marker_size * 2, 
                  marker='s', label='Depot', zorder=5)
        
        # Plot customers with size proportional to demand
        scatter = ax.scatter(customer_x, customer_y, c=demands, 
                            s=[d * 20 for d in demands], 
                            cmap='cividis', 
                            label='Customers', zorder=3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Demand', fontsize=self.font_size)
        
        # Add customer labels
        for i, (x, y, demand) in enumerate(zip(customer_x, customer_y, demands)):
            ax.annotate(f"{i+1}\n({demand})", (x, y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=self.font_size - 4, ha='left')
        
        ax.set_xlabel('X Coordinate', fontsize=self.font_size)
        ax.set_ylabel('Y Coordinate', fontsize=self.font_size)
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


def plot_routes(individual: Individual, problem: VRPProblem, 
               title: str = "VRP Solution",
               save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function to plot routes.
    
    Args:
        individual: Solution individual
        problem: VRP problem instance
        title: Plot title
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure
    """
    mapper = RouteMapper(problem)
    return mapper.plot_routes(individual, title, save_path)
