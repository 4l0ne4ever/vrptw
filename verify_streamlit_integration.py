"""
Comprehensive verification of Streamlit adaptive sizing integration.
"""
import sys
import numpy as np

def verify_imports():
    """Verify all imports work correctly."""
    print("\n" + "=" * 80)
    print("1. VERIFYING IMPORTS")
    print("=" * 80)

    try:
        # Core adaptive sizing
        from src.utils.adaptive_sizing import (
            get_adaptive_parameters,
            calculate_tightness_metrics,
            adapt_population_size,
            adapt_generation_count
        )
        print("  ✓ src.utils.adaptive_sizing imports OK")

        # Streamlit component
        from app.components.parameter_config import render_parameter_config
        print("  ✓ app.components.parameter_config imports OK")

        # Config
        from config import GA_CONFIG, ADAPTIVE_SIZING_RULES
        print("  ✓ config imports OK")

        print("\n✅ All imports successful")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_config_sync():
    """Verify config values are synced."""
    print("\n" + "=" * 80)
    print("2. VERIFYING CONFIG SYNC")
    print("=" * 80)

    try:
        from config import GA_CONFIG, ADAPTIVE_SIZING_RULES
        from app.components.parameter_config import _calculate_defaults

        # Check base config values
        checks = [
            ('GA_CONFIG.population_size', 150, GA_CONFIG.get('population_size')),
            ('GA_CONFIG.generations', 1500, GA_CONFIG.get('generations')),
            ('GA_CONFIG.elitism_rate', 0.05, GA_CONFIG.get('elitism_rate')),
            ('GA_CONFIG.mutation_prob', 0.15, GA_CONFIG.get('mutation_prob')),
            ('GA_CONFIG.tournament_size', 5, GA_CONFIG.get('tournament_size')),
            ('GA_CONFIG.use_adaptive_sizing', True, GA_CONFIG.get('use_adaptive_sizing')),
        ]

        all_pass = True
        for name, expected, actual in checks:
            if actual == expected:
                print(f"  ✓ {name}: {actual}")
            else:
                print(f"  ✗ {name}: expected {expected}, got {actual}")
                all_pass = False

        # Check Streamlit defaults for Hanoi 20 customers
        print("\n  Checking Streamlit defaults (Hanoi, 20 customers):")
        hanoi_defaults = _calculate_defaults(20, "hanoi")

        ui_checks = [
            ('population_size', 150, hanoi_defaults.get('population_size')),
            ('generations', 1500, hanoi_defaults.get('generations')),
            ('elitism_rate', 0.05, hanoi_defaults.get('elitism_rate')),
        ]

        for name, expected, actual in ui_checks:
            if actual == expected:
                print(f"  ✓ UI default {name}: {actual}")
            else:
                print(f"  ✗ UI default {name}: expected {expected}, got {actual}")
                all_pass = False

        # Check Streamlit defaults for Hanoi 100 customers
        print("\n  Checking Streamlit defaults (Hanoi, 100 customers):")
        hanoi_defaults_100 = _calculate_defaults(100, "hanoi")

        ui_checks_100 = [
            ('population_size', 200, hanoi_defaults_100.get('population_size')),
            ('generations', 2000, hanoi_defaults_100.get('generations')),
            ('elitism_rate', 0.05, hanoi_defaults_100.get('elitism_rate')),
        ]

        for name, expected, actual in ui_checks_100:
            if actual == expected:
                print(f"  ✓ UI default {name}: {actual}")
            else:
                print(f"  ✗ UI default {name}: expected {expected}, got {actual}")
                all_pass = False

        if all_pass:
            print("\n✅ Config sync verified")
        else:
            print("\n❌ Config sync failed")

        return all_pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_adaptive_sizing():
    """Verify adaptive sizing works with mock problem."""
    print("\n" + "=" * 80)
    print("3. VERIFYING ADAPTIVE SIZING LOGIC")
    print("=" * 80)

    try:
        from src.utils.adaptive_sizing import get_adaptive_parameters, calculate_tightness_metrics
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
            ("Easy (n=20, 10% tight)", MockProblem(20, 0.10), 150, 1500, 0.15),
            ("Medium (n=20, 30% tight)", MockProblem(20, 0.30), 180, 1900, 0.18),
            ("Hard (n=100, 47% tight)", MockProblem(100, 0.47), 250, 2500, 0.20),
        ]

        all_pass = True
        for desc, problem, expected_pop_min, expected_gen_min, expected_mut in scenarios:
            # Calculate metrics
            metrics = calculate_tightness_metrics(problem)

            # Get adaptive config
            base_config = GA_CONFIG.copy()
            adapted_config = get_adaptive_parameters(problem, base_config)

            pop = adapted_config['population_size']
            gen = adapted_config['generations']
            mut = adapted_config['mutation_prob']

            # Check ranges
            pop_ok = pop >= expected_pop_min - 50 and pop <= expected_pop_min + 50
            gen_ok = gen >= expected_gen_min - 300 and gen <= expected_gen_min + 300
            mut_ok = abs(mut - expected_mut) < 0.05

            status = "✓" if (pop_ok and gen_ok and mut_ok) else "✗"
            print(f"  {status} {desc}:")
            print(f"      Tight ratio: {metrics['tight_ratio']:.1%}")
            print(f"      Difficulty: {metrics['difficulty_score']:.2f}")
            print(f"      Pop={pop} (expected ~{expected_pop_min})")
            print(f"      Gen={gen} (expected ~{expected_gen_min})")
            print(f"      Mut={mut:.3f} (expected ~{expected_mut:.3f})")

            if not (pop_ok and gen_ok and mut_ok):
                all_pass = False

        if all_pass:
            print("\n✅ Adaptive sizing logic verified")
        else:
            print("\n❌ Adaptive sizing logic failed")

        return all_pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_streamlit_integration():
    """Verify Streamlit pages pass problem correctly."""
    print("\n" + "=" * 80)
    print("4. VERIFYING STREAMLIT PAGE INTEGRATION")
    print("=" * 80)

    try:
        import inspect

        # Check hanoi_mode.py
        with open('app/pages/hanoi_mode.py', 'r') as f:
            hanoi_content = f.read()

        hanoi_checks = [
            ('render_parameter_config call exists', 'render_parameter_config(' in hanoi_content),
            ('problem parameter passed', 'problem=st.session_state.hanoi_problem' in hanoi_content),
            ('dataset_type="hanoi"', 'dataset_type="hanoi"' in hanoi_content),
        ]

        print("\n  Hanoi Mode (app/pages/hanoi_mode.py):")
        all_pass = True
        for check_name, result in hanoi_checks:
            status = "✓" if result else "✗"
            print(f"    {status} {check_name}")
            if not result:
                all_pass = False

        # Check solomon_mode.py
        with open('app/pages/solomon_mode.py', 'r') as f:
            solomon_content = f.read()

        solomon_checks = [
            ('render_parameter_config call exists', 'render_parameter_config(' in solomon_content),
            ('problem parameter passed', 'problem=st.session_state.solomon_problem' in solomon_content),
            ('dataset_type="solomon"', 'dataset_type="solomon"' in solomon_content),
        ]

        print("\n  Solomon Mode (app/pages/solomon_mode.py):")
        for check_name, result in solomon_checks:
            status = "✓" if result else "✗"
            print(f"    {status} {check_name}")
            if not result:
                all_pass = False

        # Check parameter_config.py signature
        from app.components.parameter_config import render_parameter_config
        sig = inspect.signature(render_parameter_config)

        print("\n  Parameter Config Signature:")
        has_problem_param = 'problem' in sig.parameters
        print(f"    {'✓' if has_problem_param else '✗'} Has 'problem' parameter")
        if not has_problem_param:
            all_pass = False

        # Check that function handles None problem gracefully
        print("\n  Testing with None problem (should not crash):")
        try:
            # Can't actually call it without Streamlit context, but we can check the code
            with open('app/components/parameter_config.py', 'r') as f:
                param_content = f.read()

            has_none_check = 'if problem is not None:' in param_content
            print(f"    {'✓' if has_none_check else '✗'} Has None check for problem")
            if not has_none_check:
                all_pass = False
        except Exception as e:
            print(f"    ✗ Error checking None handling: {e}")
            all_pass = False

        if all_pass:
            print("\n✅ Streamlit integration verified")
        else:
            print("\n❌ Streamlit integration failed")

        return all_pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_ui_display():
    """Verify UI will display adaptive sizing info."""
    print("\n" + "=" * 80)
    print("5. VERIFYING UI DISPLAY LOGIC")
    print("=" * 80)

    try:
        with open('app/components/parameter_config.py', 'r') as f:
            content = f.read()

        checks = [
            ('Adaptive sizing import', 'from src.utils.adaptive_sizing import' in content),
            ('calculate_tightness_metrics call', 'calculate_tightness_metrics(problem)' in content),
            ('get_adaptive_parameters call', 'get_adaptive_parameters(problem' in content),
            ('Success banner', 'Adaptive Parameter Sizing Active' in content),
            ('Metrics display', 'st.metric(' in content),
            ('Tight ratio metric', 'tight_ratio' in content),
            ('Difficulty metric', 'difficulty_score' in content),
            ('Info box with recommendations', 'st.info(' in content),
            ('Evidence citation', 'Potvin (1996)' in content or 'Evidence:' in content),
            ('Error handling', 'except Exception as e:' in content),
        ]

        all_pass = True
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
            if not result:
                all_pass = False

        if all_pass:
            print("\n✅ UI display logic verified")
        else:
            print("\n❌ UI display logic incomplete")

        return all_pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verifications."""
    print("=" * 80)
    print("STREAMLIT ADAPTIVE SIZING INTEGRATION VERIFICATION")
    print("=" * 80)

    results = []

    results.append(("Imports", verify_imports()))
    results.append(("Config Sync", verify_config_sync()))
    results.append(("Adaptive Sizing Logic", verify_adaptive_sizing()))
    results.append(("Streamlit Integration", verify_streamlit_integration()))
    results.append(("UI Display Logic", verify_ui_display()))

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
        print("\nStreamlit integration is complete and correct!")
        print("\nNext steps:")
        print("1. Start Streamlit: streamlit run app/streamlit_app.py")
        print("2. Load a dataset (e.g., hanoi_lognormal_20_customers)")
        print("3. Check for '✨ Adaptive Parameter Sizing Active' banner")
        print("4. Verify recommended parameters match expected values")
        return 0
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("=" * 80)
        print("\nPlease review and fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
