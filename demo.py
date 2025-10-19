"""
VRP-GA System Demo Script
Demonstrates the system capabilities with example runs.
"""

import os
import sys
import subprocess
import time

def run_demo():
    """Run demonstration of VRP-GA system."""
    print("=" * 80)
    print("VRP-GA SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("Error: main.py not found. Please run from the project root directory.")
        return
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Demo 1: Small mockup problem
    print("\nDemo 1: Small Mockup Problem (20 customers)")
    print("-" * 50)
    cmd1 = [
        'python', 'main.py', '--generate',
        '--customers', '20',
        '--capacity', '100',
        '--generations', '500',
        '--population', '50',
        '--output', 'results/demo1'
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ Demo 1 completed successfully in {end_time - start_time:.1f} seconds")
            print("  Output saved to: results/demo1/")
        else:
            print(f"✗ Demo 1 failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Demo 1 timed out after 5 minutes")
    except Exception as e:
        print(f"✗ Demo 1 error: {e}")
    
    # Demo 2: Medium mockup problem
    print("\nDemo 2: Medium Mockup Problem (50 customers)")
    print("-" * 50)
    cmd2 = [
        'python', 'main.py', '--generate',
        '--customers', '50',
        '--capacity', '200',
        '--generations', '1000',
        '--population', '100',
        '--clustering', 'kmeans',
        '--output', 'results/demo2'
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ Demo 2 completed successfully in {end_time - start_time:.1f} seconds")
            print("  Output saved to: results/demo2/")
        else:
            print(f"✗ Demo 2 failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Demo 2 timed out after 10 minutes")
    except Exception as e:
        print(f"✗ Demo 2 error: {e}")
    
    # Demo 3: Solomon dataset (if available)
    solomon_file = 'data/solomon_dataset/C1/C101.csv'
    if os.path.exists(solomon_file):
        print("\nDemo 3: Solomon Dataset (C101)")
        print("-" * 50)
        cmd3 = [
            'python', 'main.py', '--solomon', solomon_file,
            '--generations', '1000',
            '--population', '100',
            '--output', 'results/demo3'
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd3, capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✓ Demo 3 completed successfully in {end_time - start_time:.1f} seconds")
                print("  Output saved to: results/demo3/")
            else:
                print(f"✗ Demo 3 failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("✗ Demo 3 timed out after 10 minutes")
        except Exception as e:
            print(f"✗ Demo 3 error: {e}")
    else:
        print("\nDemo 3: Solomon Dataset (C101) - SKIPPED")
        print("-" * 50)
        print("  Solomon dataset not found. Please ensure data/solomon_dataset/C1/C101.csv exists.")
    
    # Demo 4: Custom configuration
    print("\nDemo 4: Custom Configuration")
    print("-" * 50)
    cmd4 = [
        'python', 'main.py', '--generate',
        '--customers', '30',
        '--capacity', '150',
        '--generations', '800',
        '--population', '80',
        '--crossover-prob', '0.85',
        '--mutation-prob', '0.2',
        '--tournament-size', '7',
        '--elitism-rate', '0.2',
        '--clustering', 'radial',
        '--output', 'results/demo4'
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd4, capture_output=True, text=True, timeout=600)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ Demo 4 completed successfully in {end_time - start_time:.1f} seconds")
            print("  Output saved to: results/demo4/")
        else:
            print(f"✗ Demo 4 failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("✗ Demo 4 timed out after 10 minutes")
    except Exception as e:
        print(f"✗ Demo 4 error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)
    print("Check the results/ directory for generated outputs:")
    print("  - Route visualizations (PNG files)")
    print("  - Detailed reports (TXT files)")
    print("  - Data files (JSON files)")
    print("  - Convergence plots")
    print("  - Comparison charts")
    print("\nTo run individual demos:")
    print("  python main.py --generate --customers 20 --capacity 100")
    print("  python main.py --solomon data/solomon_dataset/C1/C101.csv")
    print("\nFor more options, run: python main.py --help")


def run_quick_test():
    """Run a quick test to verify the system works."""
    print("=" * 60)
    print("VRP-GA SYSTEM QUICK TEST")
    print("=" * 60)
    
    # Quick test with minimal parameters
    cmd = [
        'python', 'main.py', '--generate',
        '--customers', '10',
        '--capacity', '50',
        '--generations', '100',
        '--population', '20',
        '--no-plots',
        '--no-report',
        '--output', 'results/quick_test'
    ]
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ Quick test completed successfully in {end_time - start_time:.1f} seconds")
            print("  System is working correctly!")
            return True
        else:
            print(f"✗ Quick test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Quick test timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"✗ Quick test error: {e}")
        return False


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        success = run_quick_test()
        sys.exit(0 if success else 1)
    else:
        run_demo()
