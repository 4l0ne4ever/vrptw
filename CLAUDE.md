# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **VRP-GA System** (Vehicle Routing Problem - Genetic Algorithm) that solves delivery route optimization problems using genetic algorithms. It supports two modes:

1. **Hanoi Mode**: Real-world delivery optimization for Hanoi, Vietnam with actual GPS coordinates and Ahamove shipping cost models
2. **Solomon Mode**: Academic benchmark testing using standard Solomon VRPTW datasets with BKS (Best Known Solution) validation

The system provides both a CLI interface (`main.py`) and a Streamlit web application (`app/streamlit_app.py`).

## Development Commands

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database (optional - auto-creates on first run)
python app/database/init_db.py
```

### Running the Application

**Streamlit Web App (Primary Interface):**
```bash
# Main app
streamlit run app/streamlit_app.py

# Specific pages
streamlit run app/pages/hanoi_mode.py
streamlit run app/pages/solomon_mode.py

# Custom port
streamlit run app/streamlit_app.py --server.port 8502
```

**CLI Mode:**
```bash
# List available datasets
python main.py --list-datasets
python main.py --list-solomon
python main.py --list-mockup

# Convert Solomon datasets to JSON
python main.py --convert-solomon

# Create sample datasets
python main.py --create-samples

# Solve Solomon dataset
python main.py --solomon-dataset C101

# Solve mockup dataset
python main.py --mockup-dataset small_random

# Generate and solve mockup data
python main.py --generate --customers 50 --capacity 200

# Batch processing all Solomon datasets
python main.py --solomon-batch --generations 100 --population 50

# Custom GA parameters
python main.py --solomon-dataset C101 --generations 2000 --population 150 --crossover-prob 0.9
```

### Testing Commands
```bash
# Quick test (5 customers, < 5 seconds)
python main.py --dataset hanoi_small_5_customers --generations 100 --population 50

# Standard test (10 customers, < 30 seconds)
python main.py --dataset hanoi_medium_10_customers --generations 500 --population 100
```

## Architecture Overview

### Core Components (src/)

**Models** (`src/models/`):
- `vrp_model.py`: Core VRP problem representation (`VRPProblem`, `Customer`, `Depot`)
- `solution.py`: Solution representation (`Individual`, `Population`)

**Genetic Algorithm** (`src/algorithms/`):
- `genetic_algorithm.py`: Main GA engine with population management
- `operators.py`: Selection, crossover, mutation operators
- `fitness.py`: Fitness evaluation with constraint handling
- `decoder.py`: Chromosome-to-routes decoding
- `split.py`: Prins (2004) optimal split algorithm for route splitting
- `tw_repair.py`: Time window constraint repair operator
- `local_search.py`: 2-opt local optimization
- `nearest_neighbor.py`: Baseline heuristic

**Data Processing** (`src/data_processing/`):
- `json_loader.py`: Load datasets from JSON format
- `generator.py`: Generate mockup test data
- `distance.py`: Distance matrix calculation with traffic factors
- `constraints.py`: Constraint validation and repair
- `dataset_converter.py`: Convert Solomon datasets to JSON
- `mode_configs.py`: Mode-specific configurations (Hanoi vs Solomon)

**Evaluation** (`src/evaluation/`):
- `metrics.py`: KPI calculation (distance, cost, utilization, efficiency)
- `bks_validator.py`: Validate against Best Known Solutions
- `comparator.py`: Compare GA vs baseline solutions
- `shipping_cost.py`: Ahamove shipping cost calculation
- `result_exporter.py`: Export results to CSV/JSON

**Visualization** (`src/visualization/`):
- `reporter.py`: Generate comprehensive reports with charts

### Streamlit Application (app/)

**Pages** (`app/pages/`):
- `hanoi_mode.py`: Hanoi delivery optimization interface
- `solomon_mode.py`: Academic benchmark testing interface
- `history.py`: View optimization history
- `datasets.py`: Dataset management
- `help.py`: Documentation and help

**Components** (`app/components/`):
- UI components for data input, visualization, and results display

**Services** (`app/services/`):
- Business logic for optimization runs, data management

**Database** (`app/database/`):
- SQLAlchemy models for persisting datasets and optimization results
- Database: `data/database/vrp_app.db` (SQLite)

## Key Algorithms

### Split Algorithm (Prins 2004)
The system uses the Prins (2004) optimal split algorithm to convert giant tours (permutations of customers) into routes. This is controlled by `GA_CONFIG['use_split_algorithm']` in `config.py`.

**How it works:**
- GA evolves giant tours (customer sequences without route boundaries)
- Split algorithm uses dynamic programming to optimally partition the giant tour into routes
- Respects capacity constraints and optimizes for distance
- Time window constraints handled separately via fitness penalties

### Time Window Repair
For VRPTW problems (Solomon datasets), the system includes sophisticated time window repair:
- Configured in `GA_CONFIG['tw_repair']` section in `config.py`
- Multiple application points: after crossover/mutation, after local search, post-generation, final solution
- Soft vs hard repair modes based on violation severity
- Controlled by `max_iterations`, `violation_weight`, and threshold parameters

### Constraint Handling
**Two-tier approach:**
1. **Hard constraints** (capacity): Repaired immediately via `ConstraintHandler`
2. **Soft constraints** (time windows): Handled via fitness penalties, with optional repair operators

## Configuration

### Main Config File: `config.py`

**GA_CONFIG**: Genetic algorithm parameters
- `population_size`: 100 (standard)
- `generations`: 1000
- `crossover_prob`: 0.9
- `mutation_prob`: 0.15
- `tournament_size`: 5
- `elitism_rate`: 0.10
- `use_split_algorithm`: True (enable Prins algorithm)
- `tw_repair`: Time window repair settings

**GA_PRESETS**: Predefined configurations
- `fast`: Quick testing (pop=50, gen=500)
- `standard`: Production use (pop=100, gen=1000)
- `benchmark`: High quality (pop=100, gen=1000, more local search)

**VRP_CONFIG**: Problem-specific parameters
- `vehicle_capacity`: 200 units
- `num_vehicles`: Formula-based (ceil(n/8))
- `penalty_weight`: 5000 (constraint violation penalty)
- `traffic_factor`: 1.3 for Hanoi
- Ahamove cost model parameters

**Mode-Specific Penalty Weights:**
- Hanoi: 1200 (lighter penalties, focuses on route quality)
- Solomon: 5000 (heavier penalties, strict constraint adherence)

### Cursor Rules
From `.cursor/rules/rule1.mdc`:
- No hardcoded word lists
- No auto-documentation
- No icons unless needed
- Use magic numbers only if already proven

## Data Locations

- **Test Datasets**: `data/test_datasets/`
- **Solomon Datasets**: `data/solomon_dataset/`
- **Results**: `results/`
- **Logs**: `logs/`
- **Database**: `data/database/vrp_app.db`

### Sample Test Datasets
- `hanoi_small_5_customers.json/csv/xlsx` - Quick tests (< 5s)
- `hanoi_medium_10_customers.json/csv/xlsx` - Standard tests (< 30s)
- `hanoi_clustered_15_customers.json` - Visualization tests
- `hanoi_full_columns_10_customers.csv/xlsx` - File upload tests

## Important Implementation Notes

### Dataset Type Detection
- Set via `problem.dataset_type`: "solomon" or "hanoi"
- Affects penalty weights, initialization strategies, and repair behavior
- Solomon: Time-window focused, heavy penalties
- Hanoi: Distance-focused, lighter penalties, traffic factors

### Route Representation
- **Giant Tour**: List of customer IDs without route boundaries (e.g., `[3,1,4,2,5]`)
- **Routes**: List of routes with depot markers (e.g., `[[0,3,1,0], [0,4,2,5,0]]`)
- **Chromosome**: Giant tour representation used in GA
- Convert via `RouteDecoder.encode_routes()` and `RouteDecoder.decode_chromosome()`

### Distance Matrix Indexing
- Index 0: Depot
- Index 1 to N: Customers (where N = number of customers)
- Use `problem.get_distance(i, j)` instead of direct matrix access
- Handles ID-to-index mapping and traffic factors

### Constraint Repair Strategy
1. **During Evolution**: Capacity violations repaired in `FitnessEvaluator`
2. **Post-GA**: Final repair pass before returning solution
3. **Post-2opt**: Repair any violations introduced by local search
4. **Time Windows**: Optional repair via `TWRepairOperator` at configured points

### Streamlit Session State
Key session variables:
- `st.session_state.current_dataset`: Loaded dataset
- `st.session_state.current_problem`: VRPProblem instance
- `st.session_state.optimization_running`: Run status
- `st.session_state.optimization_results`: Latest results

## Common Development Patterns

### Adding a New Dataset
1. Place file in `data/test_datasets/` (JSON/CSV/Excel)
2. Use `JSONDatasetLoader` to load
3. Or use dataset converter: `DatasetConverter.create_mockup_dataset()`

### Running Custom Optimization
```python
from src.models.vrp_model import create_vrp_problem_from_dict
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from config import GA_CONFIG

# Load data and create problem
problem = create_vrp_problem_from_dict(data, distance_matrix)

# Configure and run GA
ga_config = GA_CONFIG.copy()
ga_config['generations'] = 500
ga = GeneticAlgorithm(problem, ga_config)
solution, evolution_data = ga.evolve()
```

### Debugging Constraint Violations
Use `--debug-constraints` flag to export detailed constraint analysis:
```bash
python main.py --dataset C101 --debug-constraints
```
Output written to `results/constraint_debug_*.txt`

### Performance Profiling
The codebase includes `pipeline_profiler` decorators on critical paths. Check logs for timing information.

## Troubleshooting

**Port already in use:**
```bash
lsof -ti:8501 | xargs kill -9
# or use different port
streamlit run app/streamlit_app.py --server.port 8502
```

**Database errors:**
```bash
python app/database/init_db.py
```

**ModuleNotFoundError:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Optimization too slow:**
- Use `fast` preset in config
- Reduce population size (50-100)
- Reduce generations (100-500)
- Use smaller dataset (5-10 customers)
- Disable local search with `--no-local-search`

**Map not displaying:**
- Check `streamlit-folium` is installed
- Verify internet connection (needed for map tiles)
- Ensure coordinates are valid

## Testing Workflow

**Quick validation:**
```bash
streamlit run app/streamlit_app.py
# Navigate to Hanoi Mode
# Upload: data/test_datasets/hanoi_small_5_customers.json
# Preset: Fast
# Run Optimization (< 5 seconds expected)
```

**Standard test:**
```bash
python main.py --solomon-dataset C101 --generations 100 --population 50
# Should complete in ~1-2 minutes
# Check BKS gap in output
```

**Batch benchmark:**
```bash
python main.py --solomon-batch --generations 1000 --population 100
# Runs all Solomon datasets
# Exports summary to results/solomon_summary_*.csv
```
