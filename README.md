# VRP-GA System

A Vehicle Routing Problem solver using Genetic Algorithm with 2-opt local search optimization.

## Features

- **Dual Data Sources**: Supports Solomon benchmark dataset and generated mockup data
- **Genetic Algorithm**: Advanced GA with tournament selection, order crossover, and adaptive mutation
- **Local Search**: 2-opt optimization for route improvement
- **Baseline Comparison**: Nearest Neighbor heuristic for performance comparison
- **Comprehensive Visualization**: Route maps, convergence plots, and KPI dashboards

## Quick Start

### Solomon Dataset
```bash
python main.py --solomon data/solomon_dataset/C1/C101.csv
```

### Generate Mockup Data
```bash
python main.py --generate --customers 50 --capacity 100
```

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
vrp-ga-system/
├── src/
│   ├── data_processing/    # Data loading and generation
│   ├── models/            # VRP model and solution classes
│   ├── algorithms/        # GA, local search, baseline
│   ├── evaluation/        # Metrics and comparison
│   └── visualization/     # Plots and reports
├── data/
│   ├── raw/              # Generated mockup data
│   ├── processed/        # Cached distance matrices
│   └── solomon_dataset/  # Benchmark data
├── results/              # Output files
└── tests/                # Unit tests
```

## Configuration

Edit `config.py` to adjust GA parameters, vehicle capacity, and other settings.
