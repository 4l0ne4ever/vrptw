"""
Plotting utilities for VRP analysis.
Creates convergence plots, comparison charts, and statistics visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional, Tuple
from src.models.solution import Individual
from src.models.vrp_model import VRPProblem
from config import VIZ_CONFIG


class Plotter:
    """Creates various plots for VRP analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize plotter.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VIZ_CONFIG.copy()
        
        # Set up matplotlib and seaborn
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.fig_size = self.config['figure_size']
        self.dpi = self.config['dpi']
        self.font_size = self.config['font_size']
    
    def plot_convergence(self, convergence_data: Dict,
                        title: str = "GA Convergence",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot GA convergence over generations.
        
        Args:
            convergence_data: Dictionary with convergence data
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        generations = convergence_data['generations']
        best_fitness = convergence_data['best_fitness']
        avg_fitness = convergence_data['avg_fitness']
        diversity = convergence_data.get('diversity', [])
        
        fig, axes = plt.subplots(2, 2, figsize=(self.fig_size[0] * 1.5, self.fig_size[1] * 1.2))
        
        # Best fitness plot
        axes[0, 0].plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
        axes[0, 0].set_xlabel('Generation', fontsize=self.font_size)
        axes[0, 0].set_ylabel('Best Fitness', fontsize=self.font_size)
        axes[0, 0].set_title('Best Fitness Evolution', fontsize=self.font_size)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Average fitness plot
        axes[0, 1].plot(generations, avg_fitness, 'g-', linewidth=2, label='Average Fitness')
        axes[0, 1].set_xlabel('Generation', fontsize=self.font_size)
        axes[0, 1].set_ylabel('Average Fitness', fontsize=self.font_size)
        axes[0, 1].set_title('Average Fitness Evolution', fontsize=self.font_size)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Fitness improvement plot
        if len(best_fitness) > 1:
            improvement = [best_fitness[i] - best_fitness[0] for i in range(len(best_fitness))]
            axes[1, 0].plot(generations, improvement, 'r-', linewidth=2, label='Fitness Improvement')
            axes[1, 0].set_xlabel('Generation', fontsize=self.font_size)
            axes[1, 0].set_ylabel('Fitness Improvement', fontsize=self.font_size)
            axes[1, 0].set_title('Fitness Improvement Over Time', fontsize=self.font_size)
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
        
        # Diversity plot
        if diversity:
            axes[1, 1].plot(generations, diversity, 'm-', linewidth=2, label='Population Diversity')
            axes[1, 1].set_xlabel('Generation', fontsize=self.font_size)
            axes[1, 1].set_ylabel('Diversity', fontsize=self.font_size)
            axes[1, 1].set_title('Population Diversity', fontsize=self.font_size)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_comparison_chart(self, comparison_data: Dict,
                            title: str = "Solution Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison chart between solutions.
        
        Args:
            comparison_data: Dictionary with comparison data
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        solution1 = comparison_data['solution1']
        solution2 = comparison_data['solution2']
        improvements = comparison_data['improvements']
        
        # Extract metrics for comparison
        metrics = ['total_distance', 'total_cost', 'efficiency_score', 'load_balance_index']
        metric_labels = ['Total Distance', 'Total Cost', 'Efficiency Score', 'Load Balance Index']
        
        solution1_values = [solution1['kpis'][metric] for metric in metrics]
        solution2_values = [solution2['kpis'][metric] for metric in metrics]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_size[0] * 2, self.fig_size[1]))
        
        # Bar chart comparison
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, solution1_values, width, 
                       label=solution1['name'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, solution2_values, width, 
                       label=solution2['name'], alpha=0.8)
        
        ax1.set_xlabel('Metrics', fontsize=self.font_size)
        ax1.set_ylabel('Values', fontsize=self.font_size)
        ax1.set_title('Solution Comparison', fontsize=self.font_size)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=self.font_size - 2)
        
        # Improvement chart
        improvement_metrics = ['distance_improvement_percent', 'cost_improvement_percent', 
                              'efficiency_improvement_percent']
        improvement_labels = ['Distance', 'Cost', 'Efficiency']
        improvement_values = [improvements[metric] for metric in improvement_metrics]
        
        colors = ['green' if val > 0 else 'red' for val in improvement_values]
        bars = ax2.bar(improvement_labels, improvement_values, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Metrics', fontsize=self.font_size)
        ax2.set_ylabel('Improvement (%)', fontsize=self.font_size)
        ax2.set_title('Improvement Analysis', fontsize=self.font_size)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, improvement_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=self.font_size - 2)
        
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_kpi_dashboard(self, kpis: Dict,
                          title: str = "KPI Dashboard",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create KPI dashboard with multiple metrics.
        
        Args:
            kpis: Dictionary with KPI values
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(self.fig_size[0] * 1.8, self.fig_size[1] * 1.2))
        
        # Distance metrics
        distance_metrics = ['total_distance', 'avg_route_distance']
        distance_values = [kpis.get(metric, 0) for metric in distance_metrics]
        distance_labels = ['Total Distance', 'Avg Route Distance']
        
        axes[0, 0].bar(distance_labels, distance_values, color=['skyblue', 'lightcoral'], alpha=0.7)
        axes[0, 0].set_title('Distance Metrics', fontsize=self.font_size)
        axes[0, 0].set_ylabel('Distance', fontsize=self.font_size - 2)
        
        # Load metrics
        load_metrics = ['avg_utilization', 'max_utilization', 'min_utilization']
        load_values = [kpis.get(metric, 0) for metric in load_metrics]
        load_labels = ['Avg', 'Max', 'Min']
        
        axes[0, 1].bar(load_labels, load_values, color=['lightgreen', 'orange', 'pink'], alpha=0.7)
        axes[0, 1].set_title('Utilization (%)', fontsize=self.font_size)
        axes[0, 1].set_ylabel('Utilization', fontsize=self.font_size - 2)
        
        # Cost metrics
        cost_metrics = ['total_cost', 'cost_per_customer', 'cost_per_route']
        cost_values = [kpis.get(metric, 0) for metric in cost_metrics]
        cost_labels = ['Total', 'Per Customer', 'Per Route']
        
        axes[0, 2].bar(cost_labels, cost_values, color=['gold', 'silver', 'bronze'], alpha=0.7)
        axes[0, 2].set_title('Cost Metrics', fontsize=self.font_size)
        axes[0, 2].set_ylabel('Cost', fontsize=self.font_size - 2)
        
        # Efficiency metrics
        efficiency_metrics = ['efficiency_score', 'feasibility_score', 'load_balance_index']
        efficiency_values = [kpis.get(metric, 0) for metric in efficiency_metrics]
        efficiency_labels = ['Efficiency', 'Feasibility', 'Balance']
        
        axes[1, 0].bar(efficiency_labels, efficiency_values, color=['purple', 'brown', 'teal'], alpha=0.7)
        axes[1, 0].set_title('Quality Metrics', fontsize=self.font_size)
        axes[1, 0].set_ylabel('Score', fontsize=self.font_size - 2)
        axes[1, 0].set_ylim(0, 1)
        
        # Route statistics
        route_stats = ['num_routes', 'avg_route_length', 'max_route_length']
        route_values = [kpis.get(metric, 0) for metric in route_stats]
        route_labels = ['Routes', 'Avg Length', 'Max Length']
        
        axes[1, 1].bar(route_labels, route_values, color=['cyan', 'magenta', 'yellow'], alpha=0.7)
        axes[1, 1].set_title('Route Statistics', fontsize=self.font_size)
        axes[1, 1].set_ylabel('Count/Length', fontsize=self.font_size - 2)
        
        # Summary text
        summary_text = f"Total Distance: {kpis.get('total_distance', 0):.2f}\n"
        summary_text += f"Routes: {kpis.get('num_routes', 0)}\n"
        summary_text += f"Customers: {kpis.get('num_customers', 0)}\n"
        summary_text += f"Execution Time: {kpis.get('execution_time', 0):.2f}s\n"
        summary_text += f"Feasible: {'Yes' if kpis.get('is_feasible', False) else 'No'}"
        
        axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=self.font_size - 2, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title('Summary', fontsize=self.font_size)
        axes[1, 2].axis('off')
        
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_statistics_table(self, statistics: Dict,
                             title: str = "Statistics Summary",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create statistics table visualization.
        
        Args:
            statistics: Dictionary with statistics
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(self.fig_size[0] * 1.2, self.fig_size[1] * 0.8))
        
        # Create table data
        table_data = []
        headers = ['Metric', 'Value', 'Unit']
        
        # Define metrics to display
        metrics_info = [
            ('Total Distance', statistics.get('total_distance', 0), 'km'),
            ('Total Cost', statistics.get('total_cost', 0), 'units'),
            ('Number of Routes', statistics.get('num_routes', 0), 'routes'),
            ('Number of Customers', statistics.get('num_customers', 0), 'customers'),
            ('Average Utilization', statistics.get('avg_utilization', 0), '%'),
            ('Load Balance Index', statistics.get('load_balance_index', 0), 'index'),
            ('Efficiency Score', statistics.get('efficiency_score', 0), 'score'),
            ('Execution Time', statistics.get('execution_time', 0), 'seconds'),
            ('Is Feasible', 'Yes' if statistics.get('is_feasible', False) else 'No', 'boolean')
        ]
        
        for metric, value, unit in metrics_info:
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            table_data.append([metric, formatted_value, unit])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(self.font_size - 2)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color alternating rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        ax.set_title(title, fontsize=self.font_size + 2, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_improvement_analysis(self, improvements: Dict,
                                 title: str = "Improvement Analysis",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot improvement analysis.
        
        Args:
            improvements: Dictionary with improvement data
            title: Plot title
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.fig_size[0] * 2, self.fig_size[1]))
        
        # Improvement percentages
        metrics = ['distance', 'cost', 'efficiency', 'load_balance']
        labels = ['Distance', 'Cost', 'Efficiency', 'Load Balance']
        percentages = [improvements[metric]['percentage'] for metric in metrics]
        
        colors = ['green' if p > 0 else 'red' for p in percentages]
        bars = ax1.bar(labels, percentages, color=colors, alpha=0.7)
        
        ax1.set_xlabel('Metrics', fontsize=self.font_size)
        ax1.set_ylabel('Improvement (%)', fontsize=self.font_size)
        ax1.set_title('Improvement Percentages', fontsize=self.font_size)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{val:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontsize=self.font_size - 2)
        
        # Improvement summary
        summary_data = []
        for metric in metrics:
            improvement = improvements[metric]
            summary_data.append([
                metric.title(),
                f"{improvement['absolute']:.2f}",
                f"{improvement['percentage']:.1f}%",
                'Yes' if improvement['is_improved'] else 'No'
            ])
        
        table = ax2.table(cellText=summary_data,
                         colLabels=['Metric', 'Absolute', 'Percentage', 'Improved'],
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(self.font_size - 2)
        table.scale(1, 1.5)
        
        # Style table
        for i in range(len(['Metric', 'Absolute', 'Percentage', 'Improved'])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Improvement Summary', fontsize=self.font_size)
        ax2.axis('off')
        
        fig.suptitle(title, fontsize=self.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


def plot_convergence(convergence_data: Dict, title: str = "GA Convergence",
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Convenience function to plot convergence.
    
    Args:
        convergence_data: Dictionary with convergence data
        title: Plot title
        save_path: Optional path to save plot
        
    Returns:
        Matplotlib figure
    """
    plotter = Plotter()
    return plotter.plot_convergence(convergence_data, title, save_path)
