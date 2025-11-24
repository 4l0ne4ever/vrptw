# VRP Strong Repair - Development Summary

## Session Overview
Date: 2025-11-24
Task: Fix Strong Repair operator to achieve 0 violations on Solomon C207 dataset (100 customers, 3 vehicles, BKS: 588.29 km)

## Architecture Context

### Pipeline Flow
```
1. Preprocessing (0.01s)
   - Distance/Time matrices
   - Neighbor lists (K=40)
   - Vidal O(1) evaluator setup

2. Genetic Algorithm (3-5 min)
   - Pure GA (no TW repair during evolution)
   - 100 generations, 100 population
   - Produces solution with 80-95 violations
   - Distance: ~600-900 km

3. Strong Repair (Target: <5s)
   - Post-GA violation fixing
   - Goal: 0 violations, maintain/improve distance
   - CURRENT BOTTLENECK

4. LNS Post-Optimization (3-5 min)
   - Fine-tune feasible solution
   - Improve distance toward BKS
```

### C207 Dataset Properties
- **Customers**: 100
- **Vehicles**: 3 (BKS uses exactly 3)
- **Vehicle Capacity**: 700 units
- **Total Demand**: 1810 units (86% utilization → capacity NOT the problem!)
- **BKS**: 588.29 km (3 vehicles, ~33 customers per vehicle)
- **Challenge**: Very tight time windows (VRPTW)

## Approaches Tried - Complete Timeline

### 1. Incremental Repair Approaches (FAILED)
Multiple attempts to fix violations one at a time through local moves.

#### 1.1 Greedy Relocate (Route Corruption Bug)
**Status**: FAILED (Fixed bug but only 1/93 violations fixed)
**Approach**: Move violated customers to first feasible position
**Problem**: Route corruption due to index shifting during removal
**Fix Applied**: Save customer before removal, insert after
**Result**: Bug fixed but still 92/93 violations remain

#### 1.2 Panic Mode (NO-OP Infinite Loop)
**Status**: FAILED (0 violations fixed, infinite loop)
**Approach**: Accept distance-increasing moves when violations high
**Problem**: Moves with 0 distance change created infinite loops
**Fix Applied**: Reject NO-OP moves (distance_delta < 0.01)
**Result**: Loop fixed but still no violations repaired

#### 1.3 Balanced Hierarchical Comparison
**Status**: FAILED (1/93 violations fixed)
**Approach**: Balance violation reduction vs distance quality
**Problem**: Algorithm stuck in local minima, can't escape
**Result**: 280s runtime, fixed 1 violation, waterbed effect

#### 1.4 Escape Mechanism (Controlled Degradation)
**Status**: FAILED (6/93 violations fixed, +52 km)
**Approach**: Allow distance increases up to 50 km to escape local minima
**Problem**: Three risks identified:
- Infinite cycling (A→B→C→A)
- Waterbed effect (fix one, break another)
- Capacity traps (block future moves)
**Result**: 250s runtime, 6 violations fixed, 87 remain

#### 1.5 Panic Swap (Two-customer Exchange)
**Status**: FAILED (6/93 violations fixed, +97 km)
**Approach**: Swap pairs of customers between routes
**Problem**: Same waterbed effect as relocate
**Result**: 280s runtime, 6 violations fixed, 87 remain

#### 1.6 Simulated Annealing (SA)
**Status**: FAILED - TOPOLOGICAL WALL HIT
**Approach**: Metropolis acceptance criterion with temperature cooling
**Parameters**:
- Initial temp: 100.0
- Cooling rate: 0.995
- Acceptance: exp(-ΔE / T) for worsening moves
**Result**:
- Runtime: 6154s (102 minutes!)
- Violations: 93 → 81 (12 fixed, 81 remain)
- Distance: +127 km
- **Conclusion**: Dead end - "trying to fix shattered vase by gluing one shard at a time while blindfolded"

**User's Critical Insight**: "We have hit the Topological Wall. STOP tuning SA."

### 2. Reconstructive Repair (CURRENT APPROACH)

#### 2.1 Core Concept: Ruin & Recreate
**Breakthrough**: Instead of fixing violations incrementally, destroy and rebuild routes from scratch.

**Strategy**:
1. **RUIN Phase**: Eject all violated customers from routes
2. **RECREATE Phase**: Rebuild using construction heuristic (Regret-2)
3. If violations > 50 (Emergency Reset): Eject ALL 100 customers and rebuild

#### 2.2 Regret-2 Insertion Heuristic
**How it works**:
```python
For each unassigned customer:
    Find all feasible insertion positions
    Calculate regret = 2nd_best_cost - best_cost

Insert customer with HIGHEST regret first
(High regret = customer with large opportunity cost if delayed)
```

**Feasibility Check**:
- ✅ Capacity constraint (demand ≤ spare capacity)
- ✅ Time window constraint (no late arrivals)
- Uses Vidal O(1) evaluator for speed

#### 2.3 Emergency Full Reset Logic
**Trigger**: If violations > 50
**Action**:
```python
# Don't just eject violated customers
# The remaining customers are in "WRONG POSITIONS"
# They block insertions for others

# Solution: Reset ALL routes to empty [0, 0]
# Rebuild all 100 customers from scratch
```

**User's Insight**: "If 78/84 customers can't be inserted, the 16 remaining are in wrong positions. Keeping them only creates obstacles (vướng víu)."

## Implementation Details

### Key Code Changes in `strong_repair.py`

#### Change 1: Emergency Reset (lines 109-137)
```python
EMERGENCY_THRESHOLD = 50

if len(violated_ids) > EMERGENCY_THRESHOLD:
    # Collect ALL customers (not just violated)
    all_customers = [c for route in current_routes
                     for c in route if c != 0]

    # Reset all routes to empty
    current_routes = [[0, 0] for _ in range(len(current_routes))]

    # Set all customers as unassigned
    violated_ids = all_customers
else:
    # Normal ruin: only eject violated customers
    for route in current_routes:
        route[:] = [c for c in route if c not in violated_ids]
```

#### Change 2: Regret-2 Loop (lines 140-197)
```python
unassigned = list(violated_ids)

while unassigned:
    best_customer = None
    max_regret = -float('inf')

    for cust_id in unassigned:
        valid_insertions = self._find_valid_insertions(
            current_routes, cust_id
        )

        if not valid_insertions:
            continue

        # Calculate Regret-2
        best_cost = valid_insertions[0][0]
        if len(valid_insertions) > 1:
            second_best = valid_insertions[1][0]
            regret = second_best - best_cost
        else:
            regret = float('inf')  # Only one position

        if regret > max_regret:
            max_regret = regret
            best_customer = cust_id
            best_insertion = valid_insertions[0]

    if best_customer is not None:
        cost, route_idx, pos = best_insertion
        current_routes[route_idx].insert(pos, best_customer)
        unassigned.remove(best_customer)
    else:
        # No feasible positions for ANY customer
        break
```

#### Change 3: Capacity Check (lines 833-842)
```python
# Fixed IndexError by using get_customer_by_id()
route_demand = 0
for cust_id in temp_route:
    if cust_id != 0:
        customer = self.problem.get_customer_by_id(cust_id)
        if customer:
            route_demand += customer.demand

if route_demand > self.problem.vehicle_capacity:
    return False, 0.0  # Capacity violation
```

## Current Problem - THE CRITICAL ISSUE

### Symptom: Hallucination Results
**Test reports**:
- ✅ Strong Repair: 500.22 km, 0 violations
- ✅ Gap: -14.97% (better than BKS!)
- ✅ "SUCCESS! Production-ready pipeline!"

**Reality (from logs)**:
```
🚨 EMERGENCY RESET: 90/100 violations (>50)
⚠️  Cannot insert remaining 84 customers (constraint deadlock)
⚠️  84 customers remain unassigned
```

**Only 16-19 out of 100 customers are routed!**

That's why:
- Distance is so low (500 km vs BKS 588 km)
- It appears "better than BKS"
- 0 violations reported (only counting routed customers)
- Solution is INVALID

### Root Cause Analysis

#### Why Only 16-19 Customers Can Be Inserted?

**Capacity is NOT the problem**:
- Total demand: 1810 units
- Total capacity (3 vehicles): 2100 units
- Utilization: 86% → 14% spare capacity

**TIME WINDOWS are the problem**:
1. Regret-2 is purely greedy
2. First 16-19 customers get inserted in "convenient" positions
3. These early insertions consume all available time slack
4. Remaining 81-84 customers can't fit ANYWHERE without violating time windows
5. Algorithm breaks - returns invalid solution with 81-84 unassigned

#### Why Greedy Regret-2 Fails

**Regret-2 only considers**:
- Distance cost (how much route distance increases)
- Opportunity cost (regret = 2nd_best - best)

**Regret-2 IGNORES**:
- Time window slack (how tight the TW constraints are)
- Time window urgency (customers with early deadlines)
- Future flexibility (will this insertion block future ones?)

**Result**: Greedy choices early on create a "constraint deadlock" where remaining customers become impossible to insert.

### Diagnostic Results
```
Quick GA (10 gen, 20 pop): 736.24 km, 97 violations
After Strong Repair:
  - Routed: 19/100 customers
  - Unassigned: 81 customers
  - Violations: 0 (only counting the 19 routed!)
```

## Proposed Solutions

### Option 1: Solomon's I1 Insertion Heuristic ⭐ RECOMMENDED
**Why**: Industry-standard construction heuristic specifically designed for VRPTW.

**Formula**:
```
cost = α1 × distance_increase
     + α2 × time_urgency
     - α3 × route_slack

where:
  distance_increase = route distance delta
  time_urgency = c_due - c_ready (prefer narrow TWs)
  route_slack = available time buffer

  α1, α2, α3 = tunable weights
```

**Advantages**:
- Considers time window urgency explicitly
- Balances distance vs feasibility
- Proven to work on Solomon benchmarks
- Used by BKS solutions

**Implementation Complexity**: Medium (replace Regret-2 with I1 formula)

### Option 2: Multiple Restarts with Random Seeding
**Approach**: Run Regret-2 insertion multiple times with different random orderings.

```python
best_solution = None
for seed in range(10):
    random.shuffle(unassigned_customers)
    solution = regret2_insertion(unassigned_customers)
    if len(solution.unassigned) < len(best_solution.unassigned):
        best_solution = solution
```

**Advantages**:
- Simple to implement
- May escape constraint deadlock by trying different orderings

**Disadvantages**:
- No guarantee of success (still greedy)
- 10x slower (but still <5s if each run is <0.5s)

### Option 3: Relaxed Insertion + Post-Repair
**Approach**: Allow TW violations during insertion, repair them afterward.

```python
# Phase 1: Insert all customers (ignore TW violations)
for cust in unassigned:
    insert_at_best_position(cust, allow_tw_violation=True)

# Phase 2: Repair TW violations with incremental moves
while violations > 0:
    apply_local_search_moves()
```

**Advantages**:
- Guarantees all customers are routed
- Separates routing from TW satisfaction

**Disadvantages**:
- May get stuck with violations (back to old problem)
- Two-phase complexity

### Option 4: Hybrid Regret-2 + Time Window Slack
**Approach**: Enhance Regret-2 to consider time window slack.

```python
# Current regret calculation
regret = second_best_cost - best_cost

# Enhanced regret calculation
slack_factor = (c_due - c_ready) / avg_tw_width
urgency_bonus = 1.0 / slack_factor  # Narrow TW = high urgency

regret_enhanced = regret + α × urgency_bonus
```

**Advantages**:
- Minimal code change
- Preserves Regret-2 logic while adding TW awareness

**Disadvantages**:
- Requires tuning α parameter
- May not fully solve the constraint deadlock

## Test Results Summary

| Approach | Runtime | Violations | Distance | Customers Routed | Status |
|----------|---------|------------|----------|------------------|--------|
| Route Corruption Fix | ~200s | 92/93 | +0 km | 100 | ❌ Local minima |
| Panic Mode | ~200s | 93/93 | +0 km | 100 | ❌ Infinite loop |
| Balanced | ~280s | 92/93 | +0 km | 100 | ❌ Local minima |
| Escape Mechanism | ~250s | 87/93 | +52 km | 100 | ❌ Waterbed |
| Panic Swap | ~280s | 87/93 | +97 km | 100 | ❌ Waterbed |
| Simulated Annealing | 6154s | 81/93 | +127 km | 100 | ❌ TOPOLOGICAL WALL |
| Greedy Regret-2 | <1s | 0 | -141 km | **16-19** | ❌ HALLUCINATION |
| Solomon I1 (greedy) | <1s | 0 | -100 km | **23** | ❌ HALLUCINATION |
| **Relaxed Insertion** | **0.2s** | **85** | **+177 km** | **100** | ✅ **PARTIAL SUCCESS** |

## Critical Insights from User

### 1. Topological Wall
> "With 84-93 violations, we've hit the Topological Wall. Incremental moves can't escape. It's like trying to fix a shattered vase by gluing one shard at a time while blindfolded."

**Implication**: Must use reconstructive approach, not incremental repair.

### 2. Wrong Positions Problem
> "If 78/84 customers can't be inserted after Emergency Reset, the remaining 16 are in WRONG POSITIONS. They block insertions. Solution: Đập đi xây lại 100% (destroy and rebuild 100%)."

**Implication**: Emergency Reset must eject ALL customers, not just violated ones.

### 3. BKS Feasibility Proof
> "BKS C207 = 3 vehicles, 588.29 km. It's definitely possible to route all 100 customers with tight constraints!"

**Implication**: The problem is solvable - our heuristic is the issue.

### 4. Hallucination Detection
> "Are you sure the result is true? -50% gap is crazy. Make sure no hallucination."

**Implication**: Always verify that ALL customers are routed, not just violation count.

## Next Steps - Prioritized Action Items

### Immediate (Critical)
1. ✅ **DONE**: Implement Emergency Full Reset
2. ✅ **DONE**: Add capacity constraint checks
3. ✅ **DONE**: Fix IndexError in capacity check
4. ❌ **BLOCKED**: Only 16-19/100 customers routed - MUST FIX

### Short Term (Solve Current Blocker)
1. **Implement Solomon's I1 Heuristic** ⭐
   - Replace Regret-2 with I1 formula
   - Add time window urgency and slack factors
   - Tune α1, α2, α3 weights

2. **Add Validation**
   - Check that all 100 customers are routed
   - Fail test if any unassigned customers
   - Report realistic BKS gaps

3. **Test I1 Implementation**
   - Quick test (10 gen GA): expect all 100 routed, 0 violations
   - Full test (100 gen GA): expect ~5-15% BKS gap

### Medium Term (Generalization)
1. Test on other Solomon datasets:
   - C101-C108: Clustered customers
   - R101-R108: Random customers
   - RC101-RC108: Mixed random-clustered
2. Verify generalization to Hanoi mode
3. Parameter tuning for better BKS gaps

### Long Term (Production)
1. Production deployment
2. Performance optimization
3. Documentation and user guides

## Code Locations

### Modified Files
- `src/optimization/strong_repair.py`
  - Lines 109-137: Emergency Reset logic
  - Lines 140-197: Regret-2 insertion loop
  - Lines 749-782: `_find_valid_insertions()`
  - Lines 816-868: `_evaluate_insertion_feasibility()`

### Test Files
- `/tmp/test_full_pipeline.py`: Full-scale production test
- `/tmp/diagnose_insertion.py`: Diagnostic for insertion failure

### Results
- `/tmp/test_final_fixed3.txt`: Latest test output (hallucination detected)

## Technical Debt

1. **Validation Gap**: Test reports success when 84 customers unassigned
2. **Greedy Heuristic**: Regret-2 doesn't consider time window urgency
3. **No Backtracking**: Once customer inserted, never moved
4. **Single-shot**: No multiple restarts or randomization

## Lessons Learned

### What Doesn't Work
1. **Incremental Repair** with 80+ violations → Topological Wall
2. **Simulated Annealing** with tight constraints → Too slow, still fails
3. **Greedy Regret-2** on VRPTW → Constraint deadlock after 16-19 insertions
4. **Accepting Worsening Moves** → Infinite loops, waterbed effects
5. **Swap/Relocate** → Can't escape local minima at this scale

### What Does Work (Partial)
1. **Emergency Full Reset** → Correctly resets all routes (implemented ✅)
2. **Regret-2 Speed** → Very fast (<1s) when it works
3. **Vidal O(1) Evaluator** → Efficient feasibility checking
4. **Relaxed Insertion** → Guarantees all 100 customers routed! ✅ (new)

### What Doesn't Work (Tested and Failed)
1. **Solomon's I1 with Strict Feasibility** → Only 23/100 customers routed
   - Time window urgency awareness doesn't help
   - Still hits constraint deadlock
2. **Any Greedy Construction Heuristic** → Fails on C207
   - Regret-2: 16-19/100 routed
   - I1: 23/100 routed
   - Root cause: Tight TWs + greedy selection = constraint deadlock

## References

### Academic Papers
- **Vidal et al. (2013)**: "Hybrid Genetic Search for the CVRP: Open-Source Implementation and SWAP* Neighborhood" → O(1) evaluator
- **Prins (2004)**: "A simple and effective evolutionary algorithm for the vehicle routing problem" → Split algorithm
- **Solomon (1987)**: "Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints" → I1 heuristic, benchmark datasets
- **Ropke & Pisinger (2006)**: "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows" → Ruin & Recreate framework

### Codebase Documentation
- `CLAUDE.md`: Project overview and development commands
- `config.py`: GA_CONFIG, penalty weights, mode configurations

## Conclusion

### Current Status: PARTIAL SUCCESS 🟡
- ✅ Emergency Reset implemented correctly
- ✅ Capacity checks added
- ✅ **BREAKTHROUGH: All 100/100 customers now routed!**
- ⚠️ 85 violations remain (reduced from 89)
- ⚠️ Distance: 832.79 km (+41.56% above BKS)

### Final Implementation: Relaxed Cheapest Insertion
After testing multiple approaches (Regret-2, Solomon I1), the solution was to **relax time window constraints during construction**:

**Strategy**:
1. Only check capacity constraints during insertion
2. Allow TW violations temporarily
3. Choose insertion position by minimum distance increase
4. Guarantees all 100 customers are routed!

**Full-Scale Test Results (100 gen, 100 pop)**:
- GA: 655.89 km, 89 violations
- Strong Repair: 832.79 km, 85 violations (0.2s)
- Runtime: <1s for Strong Repair
- **All 100/100 customers routed** ✅

### Root Cause Analysis
**Why greedy heuristics failed**:
- C207 has extremely tight time windows
- After inserting 16-23 customers greedily (with strict TW checks), remaining positions become infeasible
- Results in constraint deadlock with 77-84 unassigned customers
- Even Solomon's I1 (time-window-aware) only achieved 23/100 routed

**Why relaxed insertion works**:
- By ignoring TW constraints during construction, we avoid the constraint deadlock
- Naturally minimizes violations by choosing positions with minimum distance increase
- Guarantees all customers are routed

### Remaining Challenges
1. **85 TW violations persist** - relaxed insertion doesn't eliminate violations, only guarantees routing
2. **Distance quality** - 41.56% above BKS (acceptable for feasibility, but not optimal)
3. **Need post-repair phase** - to reduce violations from 85 → 0

### Next Steps (Future Work)
1. **Add post-repair optimization**:
   - Try 2-opt, Or-opt on completed routes
   - Use LNS destroy & repair (but on complete routes)
   - May require accepting some distance increase to fix violations

2. **Hybrid approach**:
   - Phase 1: Relaxed insertion (all customers routed)
   - Phase 2: Violation repair (reduce 85 → 0)
   - Phase 3: LNS optimization (improve distance)

3. **Alternative: Use GA with TW repair**:
   - Enable `tw_repair` in GA_CONFIG
   - May produce better initial solutions with fewer violations

### Key Learnings
1. **Greedy construction fails on tight VRPTW** - Any strict feasibility check during greedy insertion leads to constraint deadlock
2. **Relaxed construction works** - Guarantees completeness, violations can be repaired later
3. **Time window urgency doesn't help** - I1 heuristic still failed (23/100)
4. **Capacity is not the bottleneck** - 86% utilization, plenty of spare capacity
5. **BKS is achievable** - 3 vehicles, 588.29 km with 100 customers → proves it's possible

---

**Last Updated**: 2025-11-24 (Session 2)
**Status**: 🟡 PARTIAL SUCCESS - All customers routed, 85 violations remain
**Achievement**: Solved the constraint deadlock problem! All 100 customers now successfully routed.
