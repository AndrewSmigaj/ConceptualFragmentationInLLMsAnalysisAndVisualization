"""
Test if Concept MRI app can start without errors.
"""
import sys
import os

print("=" * 60)
print("Testing Concept MRI App")
print("=" * 60)

# Test 1: Check demo model exists
demo_path = "concept_mri/demos/synthetic_demo"
if os.path.exists(demo_path):
    files = os.listdir(demo_path)
    print(f"\n[OK] Demo directory exists with {len(files)} files")
    for f in files:
        print(f"  - {f}")
else:
    print("\n[ERROR] Demo directory not found. Run: python concept_mri/demos/create_simple_demo.py")

# Test 2: Check local_config.py
if os.path.exists("local_config.py"):
    print("\n[OK] local_config.py exists")
    try:
        from local_config import OPENAI_KEY
        if OPENAI_KEY:
            print("  - OPENAI_KEY is set")
        else:
            print("  - WARNING: OPENAI_KEY is empty")
    except ImportError as e:
        print(f"  - ERROR importing OPENAI_KEY: {e}")
else:
    print("\n[ERROR] local_config.py not found. Create from local_config.py.example")

# Test 3: Test imports
print("\n" + "-" * 40)
print("Testing imports...")

try:
    from concept_mri.app import app
    print("[OK] App module imports")
except Exception as e:
    print(f"[ERROR] App import failed: {e}")
    sys.exit(1)

try:
    from concept_mri.tabs.ff_networks import create_ff_networks_tab, run_llm_analysis
    print("[OK] FF networks tab imports")
except Exception as e:
    print(f"[ERROR] FF networks import failed: {e}")

try:
    from concept_mri.components.controls.clustering_panel import ClusteringPanel
    print("[OK] Clustering panel imports")
except Exception as e:
    print(f"[ERROR] Clustering panel import failed: {e}")

print("\n" + "=" * 60)
print("Import tests complete!")
print("\nTo run the app:")
print("  python concept_mri/app.py")
print("\nThen navigate to: http://localhost:8050")
print("=" * 60)