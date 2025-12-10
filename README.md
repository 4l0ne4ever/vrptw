# VRP-GA System: Vehicle Routing Problem Optimization

A comprehensive system for solving Vehicle Routing Problem with Time Windows (VRPTW) using Genetic Algorithms and advanced optimization techniques.

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone or navigate to project directory
cd optimize

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. CLI Mode (Command Line)

```bash
# Solve Solomon dataset
python main.py --mode solomon --dataset C101 --ga-population 100 --ga-generations 200

# Solve with Hanoi mockup data
python main.py --mode hanoi --dataset hanoi_lognormal_50_customers.json

# Available datasets: C101, C201, R101, R201, RC101, RC201, etc.
```

#### 2. Web UI (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

Open browser at `http://localhost:8501` for interactive interface.

## Features

- **Genetic Algorithm**: Main optimization engine with customizable parameters
- **Local Search**: 2-opt optimization for route improvement
- **Constraint Handling**: Automatic repair of capacity and time window violations
- **Distance Calculation**: OSRM API integration with Haversine fallback
- **Multi-mode Support**:
  - Solomon benchmark datasets
  - Hanoi city logistics (with real coordinates)
  - Custom mockup data
- **Visualization**: Interactive maps and performance reports
- **Shipping Cost**: Ahamove-based cost calculation
- **BKS Validation**: Compare results with Best-Known Solutions

## Project Structure

```
optimize/
├── main.py                 # CLI entry point
├── config.py              # Global configuration
├── requirements.txt       # Python dependencies
├── app/                   # Web UI (Streamlit)
│   ├── streamlit_app.py
│   ├── components/        # UI components
│   ├── pages/            # App pages (home, datasets, results)
│   └── services/         # Business logic
├── src/                  # Core algorithms
│   ├── algorithms/       # GA, local search, heuristics
│   ├── optimization/     # Repair, preprocessing
│   ├── data_processing/  # Data loading, distance calc
│   ├── evaluation/       # Metrics, validation
│   ├── models/           # VRP problem models
│   ├── visualization/    # Maps and reports
│   └── core/            # Logger, exceptions
├── data/                # Datasets (Solomon, Hanoi)
├── results/             # Output files
├── docs/                # Architecture & guides
└── tests/               # Unit tests
```

## Configuration

Edit `config.py` to customize:

```python
GA_CONFIG = {
    'population_size': 100,
    'generations': 200,
    'mutation_rate': 0.1,
    'crossover_rate': 0.8
}

VRP_CONFIG = {
    'vehicle_capacity': 1000,
    'speed': 50,  # km/h
}
```

## Key Algorithms

| Algorithm             | Purpose                                   |
| --------------------- | ----------------------------------------- |
| **Genetic Algorithm** | Population-based optimization             |
| **Split Algorithm**   | Route partition optimization (Prins 2004) |
| **2-opt**             | Local search improvement                  |
| **Nearest Neighbor**  | Fast baseline solution                    |
| **Constraint Repair** | Fix violations automatically              |

## Data Formats

### Input

- **Solomon CSV**: Standard VRPTW benchmark format
- **JSON**: Custom format with location coordinates and time windows
- **Hanoi Dataset**: Optimized for Vietnamese logistics

### Output

- Route assignments and distances
- Performance metrics (KPIs)
- Interactive maps
- PDF reports
- JSON results for integration

## Example: Solving a Problem

```bash
# 1. Using CLI
python main.py --mode solomon --dataset C101 --ga-population 150

# 2. Using Streamlit UI
streamlit run app/streamlit_app.py
# Then upload data and configure parameters in browser

# 3. Results saved to: results/
```

## Performance

- **Solomon C101**: Near-optimal solutions found in < 5 minutes
- **Hanoi 50 customers**: < 3 minutes for ~350km routes
- **Hanoi 100 customers**: < 10 minutes
- Configurable time/quality tradeoff

## Testing

```bash
pytest tests/
```

## Troubleshooting

**Issue**: OSRM API timeout

- **Solution**: Automatic fallback to Haversine distance

**Issue**: Out of memory with large datasets

- **Solution**: Reduce population size or use nearest-neighbor mode

**Issue**: Import errors

- **Solution**: Ensure virtual environment activated and all packages installed

## Documentation

- `HUONG_DAN_CHAY.md`: Vietnamese user guide
- `docs/ARCHITECTURE.md`: Detailed system architecture
- `docs/`: Additional technical documentation

## Language Support

- **Code**: English (clean, maintainable)
- **UI**: Both English and Vietnamese
- **Documentation**: Vietnamese and English

## Key Dependencies

- `numpy`, `pandas`: Data processing
- `matplotlib`, `plotly`: Visualization
- `scikit-learn`: Utilities
- `streamlit`: Web interface
- `sqlalchemy`: Database ORM
- `folium`: Maps

## Project Status

✅ Core algorithms implemented
✅ Solomon dataset support
✅ Hanoi data integration
✅ Web UI complete
✅ Multi-mode optimization
⏳ Real-time OSRM routing (optional)

## License

See LICENSE file

## Authors

4l0ne4ever
