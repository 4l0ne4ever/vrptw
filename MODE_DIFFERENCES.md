# Mode-Specific Handling: Hanoi vs Solomon

This document describes how the system handles the two distinct operational modes.

## Mode Detection

**All mode detection uses consistent pattern:**
```python
dataset_type = getattr(problem, 'dataset_type', None) or metadata.get('dataset_type', 'hanoi')
dataset_type = str(dataset_type).strip().lower()
is_solomon = dataset_type.startswith('solomon')
```

**Default:** If detection fails, defaults to 'hanoi' mode.

## Key Differences

### 1. Distance Calculation (`src/algorithms/fitness.py`)

**Solomon Mode:**
- **Pure Euclidean distance** (academic benchmarks)
- No traffic factors applied
- Formula: `distance = sqrt((x2-x1)² + (y2-y1)²)`
- Code: `route_distance += base_dist`

**Hanoi Mode with Adaptive Traffic:**
- **Real routing with time-of-day traffic**
- Adaptive traffic factor based on current time
- Peak hours (7-9AM, 5-7PM): factor = 1.8
- Normal hours: factor = 1.2
- Low hours: factor = 1.0
- Code: `segment_distance = problem.get_adaptive_distance(from_id, to_id, current_time)`

**Hanoi Mode without Adaptive Traffic:**
- **Fixed traffic multiplier**
- Uses `VRP_CONFIG['traffic_factor']` (default 1.3)
- Code: `route_distance += base_dist * traffic_factor`

### 2. Time Window Start (`src/data_processing/constraints.py` & `src/algorithms/fitness.py`)

**Solomon Mode:**
- Time windows start at **time 0**
- Example: C202 has ready_time=0, due_date=3270
- Routes start at time 0
- Code: `time_window_start = 0.0`

**Hanoi Mode:**
- Time windows start at **8:00 AM = 480 minutes**
- Reflects real-world business hours
- Routes start at 8:00 AM
- Code: `time_window_start = VRP_CONFIG.get('time_window_start', 480)`

**Fixed in:** `constraints.py:211`, `fitness.py:334,693,770`

### 3. Time Window Penalty Calculation (`src/data_processing/constraints.py`)

**Penalty Tiers (applies to both modes, but with mode-specific weights):**

```python
# Tier 1: Minor violations (< 10 time units)
penalty = 100 * violation_amount

# Tier 2: Medium violations (10-60 time units)
penalty = 1000 + 500 * (violation - 10)

# Tier 3: Large violations (60-200 time units)
penalty = 26000 + 200 * (violation - 60)

# Tier 4: Extreme violations (> 200 time units) - CAPPED WITH LOG GROWTH
penalty = 54000 + log(1 + excess) * 5000
```

**Then multiplied by mode-specific weight:**
```python
penalty *= (mode_config.time_window_penalty_hard / 1000.0)
```

**Solomon Mode Weights** (`src/data_processing/mode_configs.py`):
- `time_window_penalty_hard`: 10,000 (strict enforcement)
- `time_window_penalty_soft`: 5,000 (high - minimize violations)
- `time_window_buffer`: 0 (no flexibility)
- `allow_time_flexibility`: False

**Hanoi Mode Weights:**
- `time_window_penalty_hard`: 10,000 (high but not critical)
- `time_window_penalty_soft`: 100 (low - acceptable)
- `time_window_buffer`: 15 minutes (flexible)
- `allow_time_flexibility`: True

### 4. Penalty Capping in Fitness Function (`src/algorithms/fitness.py:227-246`)

**Solomon Mode:**
- Adaptive cap based on time window tightness
- **Loose TW** (capacity ≥ 700, e.g., C2/R2/RC2): `max_penalty = distance * 50`
- **Tight TW** (capacity < 700, e.g., C1/R1/RC1): `max_penalty = distance * 2`

**Hanoi Mode:**
- Higher flexibility cap: `max_penalty = distance * 10`

**Why different caps?**
- C2 class problems have very loose time windows (0-3270) → allows higher penalties before capping
- C1 class problems have tight time windows → must cap sooner to maintain gradient
- Hanoi has moderate flexibility → medium cap

### 5. Feasibility Acceptance (`src/data_processing/mode_configs.py`)

**Solomon Mode (Strict):**
```python
accept_near_feasible = False
feasibility_threshold = 1.0  # 100% compliance required
max_acceptable_violations = 0
max_acceptable_hard_violations = 0
```

**Hanoi Mode (Pragmatic):**
```python
accept_near_feasible = True
feasibility_threshold = 0.98  # 98% compliance OK
max_acceptable_violations = 2
max_acceptable_hard_violations = 0
```

### 6. GA Configuration

**Solomon Mode:**
- Population: 200 (larger for complex problems)
- Generations: 2500 (more iterations for quality)
- TW-aware initialization: 100% (all individuals)
- Repair intensity: "aggressive"

**Hanoi Mode:**
- Population: 150 (balanced)
- Generations: 1000 (faster)
- TW-aware initialization: 50% (diversity)
- Repair intensity: "moderate"

## Critical Fixes Applied (2025-11-22)

### Fix #1: Time Window Start Bug
**Problem:** All modes were using `time_window_start = 480`, causing Solomon routes to start 480 minutes late.

**Fix Applied:**
- `constraints.py:211` - Mode-specific start time
- `fitness.py:334,693,770` - Mode-specific start time in all violation counting functions

**Impact:** 95%+ reduction in Solomon penalties (from billions to millions)

### Fix #2: Unlimited Penalty Growth
**Problem:** Tier 3 penalty grew linearly without bound: `26000 + 1000 * violation`, causing billion-dollar penalties.

**Fix Applied:**
- `constraints.py:379-392` - Added Tier 4 with logarithmic capping
- For violations > 200: `54000 + log(1 + excess) * 5000`

**Impact:**
- Violation of 1000: Old = 966,000 → New = 87,600 (91% reduction)
- Maintains gradient for GA while preventing explosion
- Applies to BOTH modes

## Testing Requirements

To validate mode-specific handling works correctly:

**Solomon Mode Tests:**
```bash
# C1 class (tight time windows)
python main.py --solomon-dataset C101 --generations 500 --population 100

# C2 class (loose time windows)
python main.py --solomon-dataset C202 --generations 500 --population 100

# R1, R2, RC1, RC2 classes
python main.py --solomon-dataset R101 --generations 500 --population 100
```

**Hanoi Mode Tests:**
```bash
# Small dataset (5-10 customers)
streamlit run app/streamlit_app.py
# Upload: data/test_datasets/hanoi_small_5_customers.json
# Preset: Balanced

# Medium dataset (20-50 customers)
# Upload: data/test_datasets/hanoi_medium_10_customers.csv
# Preset: Balanced
```

**Expected Results:**
- **Solomon**: Distance purely Euclidean, strict feasibility, penalties proportional to violations
- **Hanoi**: Distance includes traffic factor, accepts near-feasible, lower penalties for soft violations

## Code Locations

**Mode Detection:**
- `src/algorithms/fitness.py:122,206,332,365,622,747`
- `src/data_processing/constraints.py:211`

**Distance Calculation:**
- `src/algorithms/fitness.py:306-433` (`_calculate_total_distance`)

**Time Window Validation:**
- `src/data_processing/constraints.py:175-299` (`validate_time_window_constraint`)

**Penalty Calculation:**
- `src/data_processing/constraints.py:350-402` (`_calculate_time_window_penalty`)

**Mode Configuration:**
- `src/data_processing/mode_configs.py:11-101`

## Verification Checklist

- [x] Time window start is mode-specific (0 for Solomon, 480 for Hanoi)
- [x] Distance calculation respects mode (Euclidean vs traffic-adjusted)
- [x] Penalty growth is capped to prevent explosion
- [x] Penalty weights are mode-appropriate
- [x] Feasibility criteria are mode-specific
- [x] GA parameters are mode-appropriate
- [x] All fixes are generalized (not hardcoded for specific datasets)
