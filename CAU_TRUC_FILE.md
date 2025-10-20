# CẤU TRÚC FILE VÀ THƯ MỤC HỆ THỐNG VRP-GA

## Cấu trúc thư mục tổng thể

```
/Users/duongcongthuyet/Downloads/workspace/AI /optimize/
├── 📁 data/                          # Dữ liệu
│   ├── 📁 datasets/                  # JSON datasets
│   │   ├── 📁 solomon/               # Solomon datasets (JSON)
│   │   │   ├── 📄 C101.json
│   │   │   ├── 📄 C102.json
│   │   │   ├── 📄 R101.json
│   │   │   └── ... (55 files)
│   │   └── 📁 mockup/                 # Mockup datasets (JSON)
│   │       ├── 📄 small_random.json
│   │       ├── 📄 medium_kmeans.json
│   │       └── 📄 large_kmeans.json
│   └── 📁 solomon_dataset/           # Solomon datasets gốc (CSV)
│       ├── 📁 C1/                    # Clustered instances
│       ├── 📁 C2/
│       ├── 📁 R1/                    # Random instances
│       ├── 📁 R2/
│       ├── 📁 RC1/                   # Random clustered instances
│       └── 📁 RC2/
├── 📁 src/                           # Source code
│   ├── 📁 data_processing/           # Xử lý dữ liệu
│   ├── 📁 models/                    # Mô hình dữ liệu
│   ├── 📁 algorithms/                # Thuật toán
│   ├── 📁 evaluation/                # Đánh giá và metrics
│   └── 📁 visualization/             # Visualization
├── 📁 results/                       # Kết quả
│   ├── 📄 evolution_data_*.csv       # Dữ liệu tiến hóa GA
│   ├── 📄 optimal_routes_*.txt      # Lộ trình tối ưu
│   ├── 📄 kpi_comparison_*.csv      # So sánh KPI
│   ├── 📄 solomon_summary_*.csv     # Tổng hợp Solomon
│   └── 📁 report_*/                 # Báo cáo chi tiết
│       ├── 📄 *.html                # Map Hà Nội
│       ├── 📄 *.png                 # Traditional plots
│       └── 📄 report.txt            # Báo cáo văn bản
├── 📄 main.py                       # Entry point
├── 📄 config.py                     # Cấu hình hệ thống
├── 📄 requirements.txt              # Dependencies
├── 📄 HUONG_DAN_CHAY.md             # Hướng dẫn sử dụng
├── 📄 KIEN_TRUC_WORKFLOW.md         # Kiến trúc và workflow
└── 📄 CAU_TRUC_FILE.md              # File này
```

## Chi tiết từng module

### 1. Data Processing Module (`src/data_processing/`)

```
src/data_processing/
├── 📄 json_loader.py                 # Load JSON datasets
│   ├── class JSONDatasetLoader
│   │   ├── load_dataset()
│   │   ├── load_dataset_with_distance_matrix()
│   │   └── list_available_datasets()
│   └── Auto-detect dataset type (solomon/mockup)
│
├── 📄 dataset_converter.py          # Convert Solomon CSV → JSON
│   ├── class DatasetConverter
│   │   ├── convert_solomon_to_json()
│   │   ├── create_mockup_dataset()
│   │   ├── convert_all_solomon_datasets()
│   │   └── list_datasets()
│   └── class NumpyEncoder (JSON serialization)
│
├── 📄 generator.py                  # Generate mockup data
│   ├── class MockupDataGenerator
│   │   ├── generate_customers()
│   │   ├── generate_depot()
│   │   └── export_to_csv()
│   └── Integration với Hanoi coordinates
│
├── 📄 distance.py                   # Calculate distance matrices
│   ├── class DistanceCalculator
│   │   ├── calculate_euclidean_distance()
│   │   ├── calculate_distance_matrix()
│   │   └── apply_traffic_factor()
│   └── Support for OSRM integration
│
├── 📄 constraints.py                 # Handle VRP constraints
│   ├── validate_capacity_constraint()
│   ├── validate_time_window_constraint()
│   ├── validate_vehicle_count_constraint()
│   └── calculate_constraint_violations()
│
└── 📄 enhanced_hanoi_coordinates.py  # Hanoi coordinate generation
    ├── class EnhancedHanoiCoordinateGenerator
    │   ├── generate_coordinates()
    │   ├── _is_point_in_water()
    │   ├── _generate_single_coordinate()
    │   └── get_osrm_route()
    └── Avoid water bodies, realistic locations
```

### 2. Models Module (`src/models/`)

```
src/models/
├── 📄 vrp_model.py                  # VRP problem representation
│   ├── class Customer
│   │   ├── id, x, y, demand
│   │   ├── ready_time, due_date, service_time
│   │   └── __str__(), to_dict()
│   ├── class Depot (inherits Customer)
│   └── class VRPProblem
│       ├── customers, depot, vehicle_capacity
│       ├── num_vehicles, distance_matrix
│       ├── get_distance(), get_customer_by_id()
│       ├── calculate_total_demand()
│       ├── estimate_minimum_vehicles()
│       ├── is_feasible()
│       └── get_problem_info()
│
└── 📄 solution.py                    # Solution representation
    ├── class Individual
    │   ├── chromosome, fitness, routes
    │   ├── total_distance, is_valid, penalty
    │   ├── get_route_count(), get_customer_count()
    │   ├── to_dict(), from_dict()
    │   └── __str__()
    └── class Population
        ├── individuals, best_individual
        ├── add_individual(), get_best_individual()
        ├── apply_elitism(), is_empty()
        └── __len__()
```

### 3. Algorithms Module (`src/algorithms/`)

```
src/algorithms/
├── 📄 genetic_algorithm.py          # GA engine
│   ├── class GeneticAlgorithm
│   │   ├── __init__(), initialize_population()
│   │   ├── evolve() → (best_solution, evolution_data)
│   │   ├── _create_next_generation()
│   │   ├── _update_statistics()
│   │   ├── _check_convergence()
│   │   ├── _calculate_diversity()
│   │   ├── get_statistics()
│   │   └── save_solution(), load_solution()
│   └── def run_genetic_algorithm() → (solution, stats, evolution_data)
│
├── 📄 operators.py                  # GA operators
│   ├── class SelectionOperator
│   │   ├── tournament_selection()
│   │   └── roulette_wheel_selection()
│   ├── class CrossoverOperator
│   │   ├── order_crossover()
│   │   ├── partially_mapped_crossover()
│   │   └── edge_recombination_crossover()
│   ├── class MutationOperator
│   │   ├── swap_mutation()
│   │   ├── inversion_mutation()
│   │   ├── insertion_mutation()
│   │   └── scramble_mutation()
│   └── class AdaptiveMutationOperator
│       └── adaptive_mutation_rate()
│
├── 📄 fitness.py                    # Fitness evaluation
│   ├── class FitnessEvaluator
│   │   ├── calculate_fitness()
│   │   ├── _calculate_distance_penalty()
│   │   ├── _calculate_capacity_penalty()
│   │   └── _calculate_time_window_penalty()
│   └── Fitness = 1 / (total_distance + penalty)
│
├── 📄 decoder.py                    # Chromosome → Routes decoder
│   ├── class RouteDecoder
│   │   ├── decode_chromosome()
│   │   ├── _split_into_routes()
│   │   ├── _validate_route()
│   │   └── _calculate_route_distance()
│   └── Handle capacity constraints
│
├── 📄 local_search.py               # 2-opt optimization
│   ├── class TwoOptOptimizer
│   │   ├── optimize_individual()
│   │   ├── _two_opt_route()
│   │   ├── _two_opt_inter_route()
│   │   └── _calculate_improvement()
│   └── Post-optimization step
│
└── 📄 nearest_neighbor.py           # Baseline heuristic
    ├── class NearestNeighborHeuristic
    │   ├── solve()
    │   ├── _find_nearest_customer()
    │   ├── _can_add_to_route()
    │   └── _create_route()
    └── Greedy baseline algorithm
```

### 4. Evaluation Module (`src/evaluation/`)

```
src/evaluation/
├── 📄 metrics.py                    # KPI calculation
│   ├── class KPICalculator
│   │   ├── calculate_kpis()
│   │   ├── _calculate_route_metrics()
│   │   ├── _calculate_utilization_metrics()
│   │   ├── _calculate_cost_metrics()
│   │   ├── _calculate_quality_metrics()
│   │   ├── _calculate_constraint_violations()
│   │   └── _calculate_shipping_cost()
│   └── Comprehensive KPI calculation
│
├── 📄 comparator.py                 # Solution comparison
│   ├── class SolutionComparator
│   │   ├── compare_methods()
│   │   ├── _calculate_improvements()
│   │   └── _generate_comparison_report()
│   └── GA vs NN comparison
│
├── 📄 validator.py                  # Solution validation
│   ├── validate_solution()
│   ├── validate_routes()
│   ├── validate_customers()
│   └── validate_constraints()
│
├── 📄 result_exporter.py            # Export detailed results
│   ├── class ResultExporter
│   │   ├── export_evolution_data()
│   │   ├── export_optimal_routes()
│   │   ├── export_kpi_comparison()
│   │   ├── export_sensitivity_analysis()
│   │   └── export_solomon_summary()
│   └── def export_all_results()
│
└── 📄 shipping_cost.py              # Shipping cost calculation
    ├── class ShippingCostCalculator
    │   ├── calculate_route_cost()
    │   ├── calculate_solution_cost()
    │   ├── _calculate_ahamove_cost()
    │   ├── _calculate_basic_cost()
    │   ├── generate_order_values()
    │   └── generate_waiting_times()
    ├── Ahamove pricing model
    ├── Express vs Standard service
    └── def calculate_shipping_cost_example()
```

### 5. Visualization Module (`src/visualization/`)

```
src/visualization/
├── 📄 mapper.py                     # Route mapping
│   ├── class RouteMapper
│   │   ├── plot_routes()
│   │   ├── plot_comparison()
│   │   ├── plot_route_loads()
│   │   ├── plot_customer_demands()
│   │   └── _create_route_plot()
│   └── Traditional 2D plots
│
├── 📄 plotter.py                    # Various plots
│   ├── class Plotter
│   │   ├── plot_convergence()
│   │   ├── plot_kpi_dashboard()
│   │   ├── plot_comparison_chart()
│   │   ├── plot_statistics_table()
│   │   └── plot_improvement_analysis()
│   └── Statistical plots
│
├── 📄 reporter.py                   # Report generation
│   ├── class ReportGenerator
│   │   ├── generate_comprehensive_report()
│   │   ├── generate_quick_report()
│   │   ├── _generate_visualizations()
│   │   ├── _is_hanoi_dataset()
│   │   ├── _generate_hanoi_visualizations()
│   │   ├── _generate_traditional_visualizations()
│   │   └── _generate_traditional_plots_only()
│   └── Comprehensive report generation
│
└── 📄 enhanced_hanoi_map.py         # Hanoi map visualization
    ├── class EnhancedHanoiMapVisualizer
    │   ├── create_map()
    │   ├── create_comparison_map()
    │   ├── _get_coords()
    │   ├── _get_customer_by_id()
    │   └── _decode_routes()
    ├── Folium integration
    ├── Real routes (OSRM) vs straight lines
    └── Interactive HTML maps
```

## File cấu hình và entry points

### 1. Main Entry Point (`main.py`)

```
main.py (693 lines)
├── Imports và setup
├── def create_argument_parser()      # CLI arguments
├── def list_datasets()               # List available datasets
├── def convert_solomon_datasets()    # Convert CSV → JSON
├── def create_sample_datasets()      # Create mockup datasets
├── def run_dataset_mode()            # Run with JSON dataset
├── def run_mockup_mode()             # Run with generated data
├── def run_solomon_batch()           # Batch process Solomon datasets
├── def run_single_optimization()     # Single optimization run
├── def run_optimization()            # Main optimization workflow
├── def create_depot_from_dict()      # Helper function
└── def main()                        # Entry point
```

### 2. Configuration (`config.py`)

```
config.py (56 lines)
├── GA_CONFIG                         # Genetic Algorithm settings
│   ├── population_size: 100
│   ├── generations: 500
│   ├── crossover_prob: 0.9
│   ├── mutation_prob: 0.15
│   ├── tournament_size: 5
│   ├── elitism_rate: 0.15
│   ├── convergence_threshold: 0.001
│   └── max_stagnation: 50
├── VRP_CONFIG                        # VRP problem settings
│   ├── vehicle_capacity: 200
│   ├── num_vehicles: 25
│   ├── traffic_factor: 1.0
│   ├── time_window_factor: 1.0
│   └── service_time_factor: 1.0
└── MOCKUP_CONFIG                     # Mockup data settings
    ├── n_customers: 20
    ├── clustering: 'kmeans'
    ├── seed: 42
    ├── demand_range: (5, 50)
    └── time_window_range: (0, 1000)
```

### 3. Dependencies (`requirements.txt`)

```
requirements.txt
├── numpy>=1.21.0                     # Numerical computing
├── pandas>=1.3.0                     # Data manipulation
├── matplotlib>=3.4.0                 # Plotting
├── seaborn>=0.11.0                   # Statistical plots
├── scikit-learn>=1.0.0               # Machine learning (KMeans)
├── scipy>=1.7.0                      # Scientific computing
├── pytest>=6.2.0                     # Testing framework
├── argparse                           # Command line parsing
├── json5                              # JSON handling
├── folium>=0.14.0                    # Interactive maps
└── requests>=2.25.0                  # HTTP requests (OSRM)
```

## Cấu trúc dữ liệu JSON

### 1. Dataset Format

```json
{
  "metadata": {
    "name": "C101",
    "type": "solomon",
    "description": "Solomon C1 instance",
    "customers": 100,
    "capacity": 200,
    "vehicles": 12
  },
  "depot": {
    "id": 0,
    "x": 40.0,
    "y": 50.0,
    "demand": 0,
    "ready_time": 0,
    "due_date": 1236,
    "service_time": 0
  },
  "customers": [
    {
      "id": 1,
      "x": 45.0,
      "y": 68.0,
      "demand": 10,
      "ready_time": 912,
      "due_date": 967,
      "service_time": 90
    }
  ],
  "problem_info": {
    "num_customers": 100,
    "vehicle_capacity": 200,
    "num_vehicles": 12,
    "total_demand": 1000,
    "min_vehicles_needed": 5,
    "is_feasible": true,
    "depot_location": [40.0, 50.0],
    "customer_bounds": {
      "x_min": 20.0,
      "x_max": 80.0,
      "y_min": 20.0,
      "y_max": 80.0
    }
  }
}
```

### 2. Evolution Data Format

```csv
generation,evaluated_individuals,min_fitness,max_fitness,avg_fitness,std_fitness,best_distance,avg_distance,diversity
0,99,9.995687e-06,9.999733e-06,9.997824e-06,1.593543e-09,1.666189,1.877620,8.166667
1,99,9.991699e-06,9.999733e-06,9.995274e-06,3.447479e-09,1.666189,1.850686,7.083333
```

### 3. Optimal Routes Format

```txt
=== LỘ TRÌNH TỐI ƯU GA ===

Tổng số xe sử dụng: 2
Tổng quãng đường: 1.75 km
Tổng chi phí vận chuyển: 1.75
Tổng phí giao hàng: 1,176,531 VND
Dịch vụ: express

Xe 1:
  Lộ trình: Depot → KH_1 → KH_3 → KH_5 → KH_2 → KH_8 → KH_4 → KH_7 → Depot
  Quãng đường: 0.95 km
  Tải trọng: 47.0/50 (94.0%)
  Phí giao hàng: 818,603 VND

Xe 2:
  Lộ trình: Depot → KH_9 → KH_10 → KH_6 → Depot
  Quãng đường: 0.80 km
  Tải trọng: 35.0/50 (70.0%)
  Phí giao hàng: 357,928 VND
```

## Workflow file processing

### 1. Dataset Loading Workflow

```
CSV Files (Solomon) → DatasetConverter → JSON Files → JSONDatasetLoader → VRPProblem
Mockup Config → MockupDataGenerator → JSON Files → JSONDatasetLoader → VRPProblem
```

### 2. Optimization Workflow

```
VRPProblem → GeneticAlgorithm → Individual → TwoOptOptimizer → Optimized Individual
VRPProblem → NearestNeighborHeuristic → Individual (Baseline)
```

### 3. Result Export Workflow

```
Individual + Evolution Data → ResultExporter → CSV/TXT Files
Individual + Problem → KPICalculator → KPI Metrics
Individual + Problem → ShippingCostCalculator → Shipping Cost Data
```

### 4. Visualization Workflow

```
Individual + Problem → ReportGenerator → HTML/PNG Files
Individual + Problem → EnhancedHanoiMapVisualizer → Interactive Maps
Individual + Problem → RouteMapper → Traditional Plots
```

## File naming conventions

### 1. Dataset Files

```
Solomon: C101.json, C102.json, R101.json, RC101.json, ...
Mockup: small_random.json, medium_kmeans.json, large_kmeans.json
```

### 2. Result Files

```
Evolution: evolution_data_YYYYMMDD_HHMMSS.csv
Routes: optimal_routes_YYYYMMDD_HHMMSS.txt
KPI: kpi_comparison_YYYYMMDD_HHMMSS.csv
Summary: solomon_summary_YYYYMMDD_HHMMSS.csv
```

### 3. Report Files

```
Report Directory: report_YYYYMMDD_HHMMSS/
├── ga_hanoi_map_real.html
├── ga_hanoi_map_straight.html
├── nn_hanoi_map_real.html
├── comparison_hanoi_map_real.html
├── comparison_hanoi_map_straight.html
├── ga_routes.png (for Solomon)
├── nn_routes.png (for Solomon)
├── comparison.png (for Solomon)
├── convergence.png (for Solomon)
└── report.txt
```

## Performance và scalability

### 1. File Size Estimates

```
Small Dataset (10 customers):
├── JSON: ~5KB
├── Evolution CSV: ~2KB
├── Routes TXT: ~1KB
├── KPI CSV: ~1KB
└── HTML Maps: ~50KB each

Large Dataset (100 customers):
├── JSON: ~50KB
├── Evolution CSV: ~20KB
├── Routes TXT: ~5KB
├── KPI CSV: ~2KB
└── HTML Maps: ~200KB each
```

### 2. Memory Usage

```
Population Size 100: ~10MB
Population Size 500: ~50MB
Population Size 1000: ~100MB
```

### 3. Execution Time

```
Small Dataset (10 customers):
├── GA (50 generations): ~5 seconds
├── 2-opt optimization: ~1 second
├── NN baseline: ~0.1 seconds
└── Total: ~10 seconds

Large Dataset (100 customers):
├── GA (500 generations): ~60 seconds
├── 2-opt optimization: ~10 seconds
├── NN baseline: ~1 second
└── Total: ~90 seconds
```

## Maintenance và updates

### 1. Adding New Algorithms

```
1. Create new file in src/algorithms/
2. Implement required methods
3. Update main.py to include new option
4. Add tests in tests/
5. Update documentation
```

### 2. Adding New Cost Models

```
1. Extend ShippingCostCalculator
2. Add new pricing configuration
3. Update result exporter
4. Add validation tests
5. Update documentation
```

### 3. Adding New Visualization Types

```
1. Create new visualizer in src/visualization/
2. Implement visualization methods
3. Integrate with ReportGenerator
4. Add configuration options
5. Update documentation
```

Cấu trúc file này đảm bảo tính modular, dễ bảo trì và mở rộng cho hệ thống VRP-GA.
