# CONFIG.PY REVIEW - ALIGNMENT WITH MAIN.TEX

**Date**: 2025-10-30  
**Repository**: D:\KLTN\code vrp\vrptw  
**Thesis Document**: main.tex

---

## EXECUTIVE SUMMARY

The config.py file has **SIGNIFICANT DISCREPANCIES** with the specifications in main.tex (Table 2.18). The current configuration uses **test values** that are NOT suitable for actual experiments described in the thesis.

**Status**: ❌ **NEEDS IMMEDIATE UPDATE**

---

## DETAILED COMPARISON

### 1. GA_CONFIG - GENETIC ALGORITHM PARAMETERS

#### Table Comparison:

| Parameter | main.tex (Table 2.18) | config.py (Current) | Status | Issue |
|-----------|----------------------|---------------------|--------|-------|
| population_size | **100** | **15** | ❌ WRONG | Reduced by 85% |
| generations | **1000** | **3** | ❌ CRITICAL | Reduced by 99.7% |
| crossover_prob | **0.9** | **0.9** | ✅ OK | Match |
| mutation_prob | **0.15** | **0.15** | ✅ OK | Match |
| tournament_size | **5** | **5** | ✅ OK | Match |
| elitism_rate | **10%** (0.1) | **0.15** (15%) | ⚠️ MINOR | Different value |

#### Additional Parameters in config.py (Not in main.tex):

| Parameter | config.py Value | Comment |
|-----------|----------------|---------|
| adaptive_mutation | True | Not mentioned in thesis |
| convergence_threshold | 0.001 | Reasonable, can keep |
| stagnation_limit | 50 | Reasonable, can keep |

#### ⚠️ CRITICAL ISSUES:

1. **population_size = 15**: This is TOO SMALL
   - Thesis specifies: **100**
   - Current value (15) cannot provide adequate genetic diversity
   - Will result in premature convergence

2. **generations = 3**: This is EXTREMELY SMALL
   - Thesis specifies: **1000**
   - 3 generations cannot achieve any meaningful optimization
   - This appears to be a **DEBUG/TEST value**

3. **elitism_rate = 0.15**: Slight discrepancy
   - Thesis specifies: **10%** (0.1)
   - Current: **15%** (0.15)
   - Impact: Minor, keeps more elite solutions

---

### 2. VRP_CONFIG - VRP PROBLEM PARAMETERS

#### Table Comparison:

| Parameter | main.tex (Table 2.14) | config.py | Status | Notes |
|-----------|----------------------|-----------|--------|-------|
| vehicle_capacity | **200** | **200** | ✅ OK | Match |
| num_vehicles | **25** | **25** | ✅ OK | Match |
| traffic_factor | **1.0** | **1.0** | ✅ OK | Match (no congestion) |

#### Additional Parameters in config.py:

| Parameter | Value | Alignment with Thesis |
|-----------|-------|----------------------|
| penalty_weight | 1000 | ✅ Reasonable (not explicitly mentioned) |
| depot_id | 0 | ✅ Standard (matches thesis assumption) |
| use_waiting_fee | False | ✅ Matches thesis (waiting cost = 0) |
| cod_fee_rate | 0.006 | ✅ Matches thesis (0.6%) |

**Status**: ✅ **VRP_CONFIG IS CORRECT**

---

### 3. MOCKUP_CONFIG - MOCKUP DATA GENERATION

#### Table Comparison with main.tex (Table 2.14):

| Parameter | main.tex | config.py | Status | Notes |
|-----------|----------|-----------|--------|-------|
| n_customers | 25, 50, 100 | **50** | ⚠️ PARTIAL | Only 1 value, thesis mentions 3 |
| demand_lambda | λ=7 | **7** | ✅ OK | Poisson(λ=7) |
| demand_min | 1 | **1** | ✅ OK | Match |
| demand_max | 20 | **20** | ✅ OK | Match |
| area_bounds | [0,100]×[0,100] | **(0, 100)** | ✅ OK | Match |
| clustering | kmeans/random/radial | **'kmeans'** | ✅ OK | Default is kmeans |
| n_clusters | Not specified | **5** | ✅ OK | Reasonable |
| service_time | 10 phút | **90 giây** | ❓ CHECK | 90s = 1.5 min, thesis says 10 min |
| time_window_width | 8h-20h (12h) | **200** | ❓ UNCLEAR | Units unclear |

#### ⚠️ ISSUES:

1. **service_time = 90 seconds**: 
   - Thesis Table 2.14 says: **"10 phút (trung bình thực tế tại Việt Nam)"**
   - Config has: **90 seconds = 1.5 minutes**
   - This is **6.67x SMALLER** than specified
   - **NEEDS UPDATE to 600 seconds** (10 minutes)

2. **n_customers = 50**: 
   - Thesis mentions testing with **25, 50, 100** customers
   - Config only has default of 50
   - This is OK as default, but should be configurable

---

### 4. ADDITIONAL CONFIGS (Not in main.tex)

#### VIZ_CONFIG - Visualization:
```python
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'colors': [...],
    'marker_size': 50,
    'line_width': 2,
    'font_size': 12
}
```
**Status**: ✅ OK - Not mentioned in thesis, but reasonable defaults

#### PATHS:
```python
PATHS = {
    'data_raw': 'data/raw/',
    'data_processed': 'data/processed/',
    'results': 'results/',
    'solomon_dataset': 'data/solomon_dataset/'
}
```
**Status**: ✅ OK - Good practice

---

## RECOMMENDED CHANGES

### CRITICAL - Must Fix Immediately:

```python
# Genetic Algorithm Configuration
GA_CONFIG = {
    'population_size': 100,     # CHANGE FROM 15 → 100
    'generations': 1000,        # CHANGE FROM 3 → 1000
    'crossover_prob': 0.9,      # OK
    'mutation_prob': 0.15,      # OK
    'tournament_size': 5,       # OK
    'elitism_rate': 0.10,       # CHANGE FROM 0.15 → 0.10 (10%)
    'convergence_threshold': 0.001,  # OK (keep)
    'stagnation_limit': 50      # OK (keep, though thesis says 50)
}
```

### IMPORTANT - Should Fix:

```python
# Mockup Data Generation Configuration
MOCKUP_CONFIG = {
    'n_customers': 50,
    'demand_lambda': 7,
    'demand_min': 1,
    'demand_max': 20,
    'area_bounds': (0, 100),
    'clustering': 'kmeans',
    'n_clusters': 5,
    'seed': 42,
    'service_time': 600,        # CHANGE FROM 90 → 600 (10 minutes)
    'time_window_width': 200
}
```

### OPTIONAL - Can Keep or Remove:

```python
# Remove or comment out adaptive_mutation if not implementing
GA_CONFIG = {
    # 'adaptive_mutation': True,  # Not mentioned in thesis, commented out
}
```

---

## DETAILED ANALYSIS

### 1. Why Current Config is Wrong for Experiments?

**Current GA_CONFIG (population=15, generations=3):**
- Total evaluations: 15 × 3 = **45 fitness evaluations**
- This is **INSUFFICIENT** for any meaningful optimization
- GA needs hundreds/thousands of evaluations to converge

**Correct GA_CONFIG (population=100, generations=1000):**
- Total evaluations: 100 × 1000 = **100,000 fitness evaluations**
- This aligns with thesis methodology (Section 2.5.3)
- Provides adequate time for GA to explore solution space

### 2. Impact on Results

Using current config.py values would result in:
- ❌ Poor solution quality
- ❌ No convergence
- ❌ Results NOT comparable to thesis
- ❌ Cannot reproduce thesis experiments

### 3. Why Were Small Values Used?

The current values (15, 3) appear to be **DEBUG/TEST values** for:
- Quick testing during development
- Fast iteration cycles
- Reduced computational time

These should **NEVER** be used for actual experiments.

---

## ALIGNMENT WITH THESIS SECTIONS

### Section 2.5.3 - GA Parameters (Table 2.18):
```
✅ crossover_prob: 0.9
✅ mutation_prob: 0.15
✅ tournament_size: 5
⚠️ elitism_rate: 10% (thesis) vs 15% (config)
❌ population_size: 100 (thesis) vs 15 (config)
❌ generations: 1000 (thesis) vs 3 (config)
```

### Section 2.4.2 - Hanoi Dataset Design (Table 2.14):
```
✅ vehicle_capacity: 200
✅ demand: Poisson(λ=7), [1, 20]
✅ area: [0, 100] × [0, 100]
⚠️ service_time: 10 min (thesis) vs 1.5 min (config)
```

### Section 2.3.3 - Ahamove Cost Model:
```
✅ COD rate: 0.6% (matches)
✅ waiting_cost: disabled (matches thesis assumption)
```

---

## RECOMMENDED CONFIG.PY (CORRECTED VERSION)

```python
# Configuration parameters for VRP-GA System

# Genetic Algorithm Configuration
# Based on Table 2.18 in main.tex
GA_CONFIG = {
    'population_size': 100,      # Standard for VRP problems
    'generations': 1000,         # Sufficient for convergence
    'crossover_prob': 0.9,       # High crossover rate
    'mutation_prob': 0.15,       # Moderate mutation rate
    'tournament_size': 5,        # Tournament selection
    'elitism_rate': 0.10,        # Keep top 10%
    'convergence_threshold': 0.001,
    'stagnation_limit': 50       # Stop if no improvement for 50 gens
}

# VRP Problem Configuration
# Based on Table 2.14 in main.tex
VRP_CONFIG = {
    'vehicle_capacity': 200,     # 200 units (~500kg van)
    'num_vehicles': 25,          # Default number of vehicles
    'traffic_factor': 1.0,       # No congestion (thesis assumption)
    'penalty_weight': 1000,      # Penalty for constraint violations
    'depot_id': 0,               # Depot node ID
    # Shipping cost configs (Ahamove model)
    'use_waiting_fee': False,    # Waiting cost = 0 (thesis assumption)
    'cod_fee_rate': 0.006        # 0.6% COD fee
}

# Mockup Data Generation Configuration
# Based on Table 2.14 in main.tex
MOCKUP_CONFIG = {
    'n_customers': 50,           # Default size (can use 25, 50, 100)
    'demand_lambda': 7,          # Poisson(λ=7) for demand
    'demand_min': 1,             # Minimum demand
    'demand_max': 20,            # Maximum demand
    'area_bounds': (0, 100),     # Cartesian 2D space [0,100]×[0,100]
    'clustering': 'kmeans',      # Clustering method (kmeans/random/radial)
    'n_clusters': 5,             # Number of clusters
    'seed': 42,                  # Random seed for reproducibility
    'service_time': 600,         # 10 minutes = 600 seconds (Table 2.14)
    'time_window_width': 200     # Time window width in time units
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'colors': ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', 
               '#A52A2A', '#FFC0CB', '#808080'],
    'marker_size': 50,
    'line_width': 2,
    'font_size': 12
}

# File Paths
PATHS = {
    'data_raw': 'data/raw/',
    'data_processed': 'data/processed/',
    'results': 'results/',
    'solomon_dataset': 'data/solomon_dataset/'
}
```

---

## EXECUTION TIME ESTIMATES

### Current Config (WRONG):
- Population: 15, Generations: 3
- Estimated time: **~30 seconds** (too fast to be meaningful)

### Corrected Config:
- Population: 100, Generations: 1000
- Estimated time: **~10-30 minutes** per run (depends on problem size)
- For 10 runs (thesis methodology): **~3-5 hours**

This is NORMAL and EXPECTED for GA experiments.

---

## ADDITIONAL NOTES

### 1. Parameters Not Mentioned in Thesis:

The following parameters are in config.py but not explicitly mentioned in main.tex:
- `convergence_threshold`: 0.001 - **Keep** (reasonable stopping criterion)
- `stagnation_limit`: 50 - **Keep** (matches thesis mention of 50 gens)
- `penalty_weight`: 1000 - **Keep** (standard for constraint handling)
- `n_clusters`: 5 - **Keep** (reasonable for clustering)

These are implementation details that don't conflict with thesis.

### 2. Adaptive Mutation:

Config has `adaptive_mutation: True`, but:
- Not implemented in code (uses fixed mutation rate)
- Not mentioned in thesis
- Section 2.3.2.2 explicitly says: **"Pm = 0.15 fixed"**

**Recommendation**: Remove or set to False

---

## CONCLUSION

### Summary:

| Config Section | Status | Action Required |
|---------------|--------|-----------------|
| GA_CONFIG | ❌ CRITICAL | **MUST FIX** population_size and generations |
| VRP_CONFIG | ✅ OK | No changes needed |
| MOCKUP_CONFIG | ⚠️ MINOR | Update service_time to 600 seconds |
| VIZ_CONFIG | ✅ OK | Optional, can keep |
| PATHS | ✅ OK | No changes needed |

### Priority Actions:

1. **IMMEDIATE (CRITICAL)**:
   - Change `population_size`: 15 → **100**
   - Change `generations`: 3 → **1000**
   - Change `elitism_rate`: 0.15 → **0.10**

2. **IMPORTANT**:
   - Change `service_time`: 90 → **600**
   - Remove or disable `adaptive_mutation`

3. **OPTIONAL**:
   - Add comments linking to thesis tables
   - Document why certain values were chosen

### Final Recommendation:

**Replace config.py with the corrected version above** before running any experiments for the thesis. The current config.py appears to contain debug/test values that are unsuitable for research purposes.

---

**Last Updated**: 2025-10-30 16:43  
**Reviewed By**: AI Assistant  
**Status**: ❌ REQUIRES IMMEDIATE UPDATE
