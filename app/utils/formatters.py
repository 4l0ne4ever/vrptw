"""
Data formatting functions.
"""

from typing import Dict, List
import pandas as pd


def format_dataset_for_display(data_dict: Dict) -> pd.DataFrame:
    """
    Format dataset for display in table.
    
    Args:
        data_dict: Dataset data dictionary
        
    Returns:
        DataFrame with formatted data
    """
    customers_data = []
    
    # Add depot
    depot = data_dict['depot']
    customers_data.append({
        'ID': depot['id'],
        'Type': 'Depot',
        'X (Lon)': f"{depot['x']:.6f}",
        'Y (Lat)': f"{depot['y']:.6f}",
        'Demand': depot['demand']
    })
    
    # Add customers
    for customer in data_dict['customers']:
        customers_data.append({
            'ID': customer['id'],
            'Type': 'Customer',
            'X (Lon)': f"{customer['x']:.6f}",
            'Y (Lat)': f"{customer['y']:.6f}",
            'Demand': customer['demand']
        })
    
    return pd.DataFrame(customers_data)


def create_csv_template() -> str:
    """
    Create CSV template for download.
    
    Returns:
        CSV template as string
    """
    template = """id,x,y,demand,ready_time,due_date,service_time
0,105.8542,21.0285,0,0,1000,0
1,105.8400,21.0200,10,0,1000,10
2,105.8500,21.0300,15,0,1000,10
3,105.8600,21.0250,20,0,1000,10"""
    
    return template

