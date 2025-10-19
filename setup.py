"""
VRP-GA System Setup Script
Sets up the project environment and runs initial tests.
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        # Install from requirements.txt
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Dependencies installed successfully")
            return True
        else:
            print(f"✗ Failed to install dependencies: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = [
        'data/raw',
        'data/processed',
        'results',
        'tests',
        'notebooks',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def run_tests():
    """Run unit tests."""
    print("\nRunning unit tests...")
    
    try:
        result = subprocess.run([
            sys.executable, 'tests/run_tests.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All tests passed")
            return True
        else:
            print(f"✗ Tests failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running tests: {e}")
        return False

def run_quick_demo():
    """Run a quick demo to verify everything works."""
    print("\nRunning quick demo...")
    
    try:
        result = subprocess.run([
            sys.executable, 'demo.py', '--quick'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Quick demo completed successfully")
            return True
        else:
            print(f"✗ Quick demo failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running quick demo: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 80)
    print("VRP-GA SYSTEM SETUP")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\nWarning: Some dependencies may not have installed correctly.")
        print("You may need to install them manually.")
    
    # Run tests
    if not run_tests():
        print("\nWarning: Some tests failed.")
        print("The system may still work, but there might be issues.")
    
    # Run quick demo
    if not run_quick_demo():
        print("\nWarning: Quick demo failed.")
        print("Please check the error messages above.")
    
    print("\n" + "=" * 80)
    print("SETUP COMPLETE")
    print("=" * 80)
    print("The VRP-GA system has been set up.")
    print("\nNext steps:")
    print("1. Run a full demo: python demo.py")
    print("2. Try the system: python main.py --generate --customers 20")
    print("3. Check the README.md for more examples")
    print("\nFor help: python main.py --help")

if __name__ == '__main__':
    main()
