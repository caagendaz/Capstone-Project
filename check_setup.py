"""
Pre-flight Check Script

Run this before starting the main pipeline to verify all setup requirements.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version >= 3.8"""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'requests': 'requests'
    }
    
    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - Run: pip install {package}")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check if required directories exist"""
    print("\nChecking project directories...", end=" ")
    
    required_dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "models",
        "results/figures",
        "results/metrics",
        "src"
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"\n  ✗ Missing: {dir_path}")
            all_ok = False
    
    if all_ok:
        print("✓")
    
    return all_ok

def test_bvbrc_api():
    """Test BVBRC API connectivity"""
    print("\nTesting BVBRC API...", end=" ")
    
    try:
        import requests
        response = requests.get(
            "https://www.bv-brc.org/api/genome/",
            params={"eq(genome_id,511145.183)": "", "limit": 1},
            timeout=10
        )
        
        if response.ok:
            print("✓")
            return True
        else:
            print(f"✗ Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Check internet connection")
        return False

def main():
    """Run all pre-flight checks"""
    print("="*60)
    print("PRE-FLIGHT CHECKS")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages),
        ("Project Directories", check_directories),
        ("BVBRC API", test_bvbrc_api)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  python main.py --steps all --antibiotic CIPROFLOXACIN")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help, see:")
        print("  - SETUP.md (API credentials setup)")
        print("  - QUICKSTART.md (installation guide)")
        print("  - README.md (full documentation)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
