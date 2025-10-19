# VRP-GA System

A comprehensive Vehicle Routing Problem (VRP) solver using Genetic Algorithm with 2-opt local search optimization.

## Features

- **Dual Data Sources**: Supports Solomon benchmark dataset and generated mockup data
- **Advanced Genetic Algorithm**: Tournament selection, order crossover, adaptive mutation
- **Local Search Optimization**: 2-opt intra-route and inter-route improvements
- **Baseline Comparison**: Nearest Neighbor heuristic for performance evaluation
- **Comprehensive Evaluation**: KPI metrics, constraint validation, solution comparison
- **Rich Visualizations**: Route maps, convergence plots, comparison charts, KPI dashboards
- **Detailed Reporting**: Comprehensive reports with statistics and analysis

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd vrp-ga-system

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Solve Solomon Dataset

```bash
python main.py --solomon data/solomon_dataset/C1/C101.csv
```

#### Generate and Solve Mockup Data

```bash
python main.py --generate --customers 50 --capacity 200
```

#### Custom Parameters

```bash
python main.py --solomon data/solomon_dataset/C1/C101.csv \
               --generations 2000 \
               --population 150 \
               --crossover-prob 0.85 \
               --mutation-prob 0.2
```

## Command Line Options

### Data Source

- `--solomon <file>`: Use Solomon dataset CSV file
- `--generate`: Generate mockup data

### Mockup Generation

- `--customers <n>`: Number of customers (default: 50)
- `--capacity <c>`: Vehicle capacity (default: 200)
- `--clustering <method>`: Clustering method (random, kmeans, radial)
- `--area <name>`: Area name for mockup data

### GA Parameters

- `--generations <n>`: Number of generations (default: 1000)
- `--population <n>`: Population size (default: 100)
- `--crossover-prob <p>`: Crossover probability (default: 0.9)
- `--mutation-prob <p>`: Mutation probability (default: 0.15)
- `--tournament-size <n>`: Tournament size (default: 5)
- `--elitism-rate <r>`: Elitism rate (default: 0.15)

### VRP Parameters

- `--vehicles <n>`: Number of vehicles (auto-determined if not specified)
- `--traffic-factor <f>`: Traffic factor for distance calculation

### Output Options

- `--output <dir>`: Output directory (default: results)
- `--no-plots`: Skip generating plots
- `--no-report`: Skip generating report
- `--save-solution`: Save solution data to JSON

### Algorithm Options

- `--no-local-search`: Skip 2-opt local search optimization
- `--no-baseline`: Skip Nearest Neighbor baseline

### Debug Options

- `--verbose`: Verbose output
- `--seed <n>`: Random seed for reproducibility

## Project Structure

```
vrp-ga-system/
├── src/
│   ├── data_processing/    # Data loading and generation
│   │   ├── loader.py       # Solomon dataset loader
│   │   ├── generator.py    # Mockup data generator
│   │   ├── distance.py     # Distance calculation
│   │   └── constraints.py # Constraint handling
│   ├── models/            # VRP model and solution classes
│   │   ├── vrp_model.py   # VRP problem model
│   │   └── solution.py    # Solution representation
│   ├── algorithms/        # GA, local search, baseline
│   │   ├── genetic_algorithm.py  # Main GA engine
│   │   ├── operators.py          # GA operators
│   │   ├── fitness.py           # Fitness evaluation
│   │   ├── decoder.py           # Route decoding
│   │   ├── local_search.py      # 2-opt optimization
│   │   └── nearest_neighbor.py  # Baseline heuristic
│   ├── evaluation/        # Metrics and comparison
│   │   ├── metrics.py     # KPI calculation
│   │   ├── comparator.py  # Solution comparison
│   │   └── validator.py   # Solution validation
│   └── visualization/     # Plots and reports
│       ├── mapper.py      # Route mapping
│       ├── plotter.py     # Plotting utilities
│       └── reporter.py    # Report generation
├── data/
│   ├── raw/              # Generated mockup data
│   ├── processed/        # Cached distance matrices
│   └── solomon_dataset/  # Benchmark data
├── tests/                # Unit tests
├── results/              # Output files
├── config.py             # Configuration parameters
├── main.py              # Main application
└── requirements.txt     # Python dependencies
```

## Configuration

Edit `config.py` to adjust system parameters:

```python
# GA Configuration
GA_CONFIG = {
    'population_size': 100,
    'generations': 1000,
    'crossover_prob': 0.9,
    'mutation_prob': 0.15,
    'tournament_size': 5,
    'elitism_rate': 0.15
}

# VRP Configuration
VRP_CONFIG = {
    'vehicle_capacity': 200,
    'num_vehicles': 25,
    'traffic_factor': 1.0
}
```

## Examples

### Example 1: Small Problem

```bash
python main.py --generate --customers 20 --capacity 100 --generations 500
```

### Example 2: Large Problem

```bash
python main.py --generate --customers 100 --capacity 200 --generations 2000 --population 200
```

### Example 3: Solomon Benchmark

```bash
python main.py --solomon data/solomon_dataset/C1/C101.csv --generations 1500
```

### Example 4: Custom Configuration

```bash
python main.py --generate --customers 50 \
               --generations 1000 \
               --population 150 \
               --crossover-prob 0.85 \
               --mutation-prob 0.2 \
               --tournament-size 7 \
               --elitism-rate 0.2 \
               --traffic-factor 1.2
```

## Output

The system generates comprehensive output including:

### Visualizations

- Route maps showing vehicle paths
- Convergence plots showing GA progress
- Comparison charts between GA and baseline
- KPI dashboards with key metrics
- Load distribution plots

### Reports

- Detailed text report with statistics
- JSON data files for further analysis
- Solution validation results
- Performance comparisons

### Example Output Structure

```
results/
├── report_20231201_143022/
│   ├── report.txt              # Detailed text report
│   ├── report_data.json        # Complete data
│   ├── ga_routes.png           # GA route visualization
│   ├── nn_routes.png           # NN route visualization
│   ├── comparison.png          # Route comparison
│   ├── convergence.png         # GA convergence plot
│   ├── ga_dashboard.png         # GA KPI dashboard
│   ├── nn_dashboard.png        # NN KPI dashboard
│   ├── comparison_chart.png    # Metrics comparison
│   └── improvements.png        # Improvement analysis
```

## Testing

Run the test suite:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test file
python tests/test_algorithms.py
```

## Performance

The system is designed for performance with:

- Efficient distance matrix caching
- Optimized genetic operators
- Parallel processing capabilities
- Memory-efficient data structures

Typical performance on modern hardware:

- 50 customers: ~10-30 seconds
- 100 customers: ~1-3 minutes
- 200 customers: ~5-15 minutes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{vrp_ga_system,
  title={VRP-GA System: Vehicle Routing Problem Solver using Genetic Algorithm},
  author={Your Name},
  year={2023},
  url={https://github.com/yourusername/vrp-ga-system}
}
```

## Support

For questions, issues, or contributions, please:

- Open an issue on GitHub
- Contact the maintainers
- Check the documentation

## Changelog

### Version 1.0.0

- Initial release
- Complete VRP-GA implementation
- Solomon dataset support
- Mockup data generation
- Comprehensive visualization
- Detailed reporting system
