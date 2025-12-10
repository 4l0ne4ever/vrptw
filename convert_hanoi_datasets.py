"""
Convert Hanoi CSV datasets to JSON format.
"""
import json
import pandas as pd
import os

def convert_hanoi_csv_to_json(csv_file: str, output_file: str = None):
    """Convert Hanoi CSV file to JSON format."""

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Extract depot (first row, id=0)
    depot_row = df[df['id'] == 0].iloc[0]
    depot = {
        "id": int(depot_row['id']),
        "x": float(depot_row['x']),
        "y": float(depot_row['y']),
        "demand": float(depot_row['demand']),
        "ready_time": float(depot_row['ready_time']),
        "due_date": float(depot_row['due_date']),
        "service_time": float(depot_row['service_time'])
    }

    # Extract customers (all rows except depot)
    customer_rows = df[df['id'] != 0]
    customers = []
    for _, row in customer_rows.iterrows():
        customer = {
            "id": int(row['id']),
            "x": float(row['x']),
            "y": float(row['y']),
            "demand": float(row['demand']),
            "ready_time": float(row['ready_time']),
            "due_date": float(row['due_date']),
            "service_time": float(row['service_time'])
        }
        customers.append(customer)

    # Get vehicle info from first row
    vehicle_capacity = int(depot_row['vehicle_capacity'])
    num_vehicles = int(depot_row['num_vehicles'])

    # Get base name for metadata
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Determine distribution type from filename
    if 'lognormal_relaxed' in base_name:
        distribution = "log-normal (relaxed)"
    elif 'lognormal' in base_name:
        distribution = "log-normal"
    elif 'normal' in base_name:
        distribution = "normal"
    else:
        distribution = "uniform"

    # Create JSON structure
    json_data = {
        "metadata": {
            "name": base_name,
            "source": csv_file,
            "format": "hanoi_csv",
            "description": f"Hanoi VRP dataset with {distribution} time window distribution",
            "num_customers": len(customers),
            "vehicle_capacity": vehicle_capacity,
            "num_vehicles": num_vehicles,
            "distribution": distribution
        },
        "depot": depot,
        "customers": customers,
        "problem_config": {
            "vehicle_capacity": vehicle_capacity,
            "num_vehicles": num_vehicles,
            "traffic_factor": 1.0,
            "penalty_weight": 1000
        }
    }

    # Generate output filename if not provided
    if output_file is None:
        output_file = csv_file.replace('.csv', '.json')

    # Save JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Converted: {csv_file} -> {output_file}")
    return output_file

def convert_all_hanoi_datasets(directory: str = "data/test_datasets"):
    """Convert all Hanoi CSV files in directory to JSON."""

    converted_count = 0
    csv_files = [
        # NORMAL distribution
        "hanoi_normal_20_customers.csv",
        "hanoi_normal_50_customers.csv",
        "hanoi_normal_100_customers.csv",

        # LOG-NORMAL distribution
        "hanoi_lognormal_20_customers.csv",
        "hanoi_lognormal_50_customers.csv",
        "hanoi_lognormal_100_customers.csv",

        # LOG-NORMAL RELAXED distribution
        "hanoi_lognormal_relaxed_20_customers.csv",
        "hanoi_lognormal_relaxed_50_customers.csv",
        "hanoi_lognormal_relaxed_100_customers.csv",

        # UNIFORM distribution (existing)
        "hanoi_medium_20_customers.csv",
        "hanoi_large_50_customers.csv",
        "hanoi_xlarge_100_customers.csv"
    ]

    for csv_file in csv_files:
        csv_path = os.path.join(directory, csv_file)
        if os.path.exists(csv_path):
            try:
                convert_hanoi_csv_to_json(csv_path)
                converted_count += 1
            except Exception as e:
                print(f"Error converting {csv_file}: {e}")
        else:
            print(f"File not found: {csv_path}")

    print(f"\nConverted {converted_count} datasets to JSON format")

if __name__ == "__main__":
    convert_all_hanoi_datasets()
