"""
Comprehensive verification of implementation changes.
Run this script to verify all changes are correctly implemented.
"""
import sys
import numpy as np
from typing import Dict, List

def verify_config_changes() -> bool:
    """Verify config.py changes."""
    print("\n" + "=" * 80)
    print("1. VERIFYING CONFIG CHANGES")
    print("=" * 80)

    try:
        from config import GA_CONFIG, ADAPTIVE_SIZING_RULES

        # Check base config values
        checks = [
            ('use_adaptive_sizing', True, GA_CONFIG.get('use_adaptive_sizing')),
            ('population_size', 150, GA_CONFIG.get('population_size')),
            ('generations', 1500, GA_CONFIG.get('generations')),
            ('elitism_rate', 0.05, GA_CONFIG.get('elitism_rate')),
            ('adaptive_mutation', True, GA_CONFIG.get('adaptive_mutation')),
        ]

        all_pass = True
        for name, expected, actual in checks:
            if actual == expected:
                print(f"  ✓ {name}: {actual}")
            else:
                print(f"  ✗ {name}: expected {expected}, got {actual}")
                all_pass = False

        # Check ADAPTIVE_SIZING_RULES structure
        required_sections = ['population', 'generations', 'mutation', 'tournament']
        for section in required_sections:
            if section in ADAPTIVE_SIZING_RULES:
                print(f"  ✓ ADAPTIVE_SIZING_RULES['{section}'] exists")
            else:
                print(f"  ✗ ADAPTIVE_SIZING_RULES['{section}'] missing")
                all_pass = False

        if all_pass:
            print("\n✅ Config changes verified")
        else:
            print("\n❌ Config verification failed")

        return all_pass

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_adaptive_sizing() -> bool:
    """Verify adaptive sizing module."""
    print("\n" + "=" * 80)
    print("2. VERIFYING ADAPTIVE SIZING MODULE")
    print("=" * 80)

    try:
        from src.utils.adaptive_sizing import (
            get_adaptive_parameters,
            calculate_tightness_metrics,
            adapt_population_size,
            adapt_generation_count
        )
        from config import GA_CONFIG

        # Create mock problem
        class MockCustomer:
            def __init__(self, ready_time, due_date):
                self.ready_time = ready_time
                self.due_date = due_date

        class MockProblem:
            def __init__(self, n, tight_pct):
                self.customers = []
                for i in range(n):
                    tw_width = 70 if i < int(n * tight_pct) else 150
                    ready = 500 + i * 10
                    self.customers.append(MockCustomer(ready, ready + tw_width))

        # Test scenarios
        scenarios = [
            ("Easy (n=20, 10% tight)", MockProblem(20, 0.10), 150, 1500),
            ("Medium (n=20, 30% tight)", MockProblem(20, 0.30), 180, 1900),
            ("Hard (n=100, 47% tight)", MockProblem(100, 0.47), 250, 2500),
        ]

        all_pass = True
        for desc, problem, expected_pop_min, expected_gen_min in scenarios:
            config = get_adaptive_parameters(problem, GA_CONFIG.copy())
            pop = config['population_size']
            gen = config['generations']

            pop_ok = pop >= expected_pop_min - 30 and pop <= expected_pop_min + 50
            gen_ok = gen >= expected_gen_min - 200 and gen <= expected_gen_min + 200

            status = "✓" if (pop_ok and gen_ok) else "✗"
            print(f"  {status} {desc}: Pop={pop}, Gen={gen}")

            if not (pop_ok and gen_ok):
                all_pass = False

        if all_pass:
            print("\n✅ Adaptive sizing verified")
        else:
            print("\n❌ Adaptive sizing verification failed")

        return all_pass

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_two_opt_fix() -> bool:
    """Verify 2-opt TW-aware fix."""
    print("\n" + "=" * 80)
    print("3. VERIFYING 2-OPT TW-AWARE FIX")
    print("=" * 80)

    try:
        from src.algorithms.local_search import TwoOptOptimizer
        import inspect

        # Check method exists
        if not hasattr(TwoOptOptimizer, '_check_route_tw_feasibility'):
            print("  ✗ _check_route_tw_feasibility method missing")
            return False
        print("  ✓ _check_route_tw_feasibility method exists")

        # Check integration in _two_opt_single_route
        source = inspect.getsource(TwoOptOptimizer._two_opt_single_route)

        if '_check_route_tw_feasibility' not in source:
            print("  ✗ TW check not integrated in 2-opt")
            return False
        print("  ✓ TW check integrated in 2-opt")

        if 'self._check_route_tw_feasibility(new_route)' not in source:
            print("  ✗ TW check not called correctly")
            return False
        print("  ✓ TW check called correctly in swap logic")

        # Check it's in the right conditional
        if 'self._check_route_capacity(new_route) and' in source and \
           '_check_route_tw_feasibility(new_route)' in source:
            print("  ✓ TW check combined with capacity check")
        else:
            print("  ✗ TW check not properly combined with other checks")
            return False

        print("\n✅ 2-opt TW-aware fix verified")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_ga_integration() -> bool:
    """Verify GA integration."""
    print("\n" + "=" * 80)
    print("4. VERIFYING GA INTEGRATION")
    print("=" * 80)

    try:
        from src.algorithms.genetic_algorithm import GeneticAlgorithm
        from src.models.vrp_model import VRPProblem, Customer, Depot
        import inspect

        # Check adaptive sizing is in __init__
        source = inspect.getsource(GeneticAlgorithm.__init__)

        if 'use_adaptive_sizing' not in source:
            print("  ✗ Adaptive sizing check not in GA __init__")
            return False
        print("  ✓ Adaptive sizing check in GA __init__")

        if 'get_adaptive_parameters' not in source:
            print("  ✗ get_adaptive_parameters not called")
            return False
        print("  ✓ get_adaptive_parameters called")

        # Test actual initialization
        depot = Depot(id=0, x=0, y=0, ready_time=0, due_date=1440, service_time=0)
        customers = []
        for i in range(20):
            tw_width = 70 if i < 6 else 150
            ready = 500 + i * 20
            customers.append(Customer(
                id=i+1, x=i*5, y=i*5, demand=5,
                ready_time=ready, due_date=ready+tw_width, service_time=10
            ))

        problem = VRPProblem(depot=depot, customers=customers,
                           vehicle_capacity=100, num_vehicles=4)
        problem.distance_matrix = np.random.uniform(1, 50, (21, 21))
        np.fill_diagonal(problem.distance_matrix, 0)

        from config import GA_CONFIG
        config = GA_CONFIG.copy()
        config['use_adaptive_sizing'] = True

        ga = GeneticAlgorithm(problem, config)

        if ga.config.get('_adapted', False):
            print("  ✓ Config was adapted")
        else:
            print("  ✗ Config was not adapted")
            return False

        if ga.config['population_size'] > 150:
            print(f"  ✓ Population adapted to {ga.config['population_size']}")
        else:
            print("  ✗ Population not adapted")
            return False

        print("\n✅ GA integration verified")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_no_conflicts() -> bool:
    """Verify no conflicts or circular dependencies."""
    print("\n" + "=" * 80)
    print("5. VERIFYING NO CONFLICTS")
    print("=" * 80)

    try:
        # Test different import orders
        import sys

        # Clear modules
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith('src.')]
        for mod in modules_to_clear:
            del sys.modules[mod]

        # Import order 1
        from src.utils.adaptive_sizing import get_adaptive_parameters
        from src.algorithms.genetic_algorithm import GeneticAlgorithm
        from src.algorithms.local_search import TwoOptOptimizer
        print("  ✓ Import order 1: OK")

        # Clear and try reverse
        modules_to_clear = [k for k in sys.modules.keys() if k.startswith('src.')]
        for mod in modules_to_clear:
            del sys.modules[mod]

        from src.algorithms.local_search import TwoOptOptimizer
        from src.algorithms.genetic_algorithm import GeneticAlgorithm
        from src.utils.adaptive_sizing import get_adaptive_parameters
        print("  ✓ Import order 2: OK")

        print("\n✅ No conflicts detected")
        return True

    except Exception as e:
        print(f"\n❌ Conflict detected: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verifications."""
    print("=" * 80)
    print("COMPREHENSIVE IMPLEMENTATION VERIFICATION")
    print("=" * 80)

    results = []

    results.append(("Config Changes", verify_config_changes()))
    results.append(("Adaptive Sizing", verify_adaptive_sizing()))
    results.append(("2-opt TW Fix", verify_two_opt_fix()))
    results.append(("GA Integration", verify_ga_integration()))
    results.append(("No Conflicts", verify_no_conflicts()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED")
        print("=" * 80)
        print("\nImplementation is correct and ready for testing!")
        return 0
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("=" * 80)
        print("\nPlease review and fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
