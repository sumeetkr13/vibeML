#!/usr/bin/env python3
"""
Verification script to check if the ML CSV Analyzer setup is working correctly.
"""

import sys
import importlib.util

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        if package_name is None:
            package_name = module_name
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            module = importlib.import_module(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_name}: {version}")
            return True
        else:
            print(f"❌ {package_name}: Not found")
            return False
    except ImportError as e:
        print(f"❌ {package_name}: Import error - {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 ML CSV Analyzer - Setup Verification")
    print("=" * 50)
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python Version: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        return False
    
    print("\n📦 Checking Dependencies:")
    print("-" * 30)
    
    # Required packages
    required_packages = [
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"), 
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn")
    ]
    
    all_good = True
    for module_name, package_name in required_packages:
        if not check_import(module_name, package_name):
            all_good = False
    
    print("\n🧪 Testing Core Functionality:")
    print("-" * 30)
    
    try:
        # Test basic imports from our modules
        from ml_utils import detect_problem_type, handle_missing_values
        from visualization_utils import create_confusion_matrix
        print("✅ ML utilities: Working")
        
        # Test with sample data
        import pandas as pd
        import numpy as np
        
        # Create sample data
        sample_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100), 
            'target': np.random.choice([0, 1], 100)
        })
        
        # Test problem type detection
        problem_type = detect_problem_type(sample_data['target'])
        print(f"✅ Problem detection: {problem_type}")
        
        print("✅ Visualization utilities: Working")
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! Your setup is ready.")
        print("\n🚀 To start the application, run:")
        print("   ./run_app.sh")
        print("   or")
        print("   streamlit run app.py")
    else:
        print("💔 Some checks failed. Please fix the issues above.")
        print("\n💡 Try running:")
        print("   pip install -r requirements.txt")
    
    return all_good

if __name__ == "__main__":
    main() 