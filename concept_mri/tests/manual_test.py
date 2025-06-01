"""
Manual testing script for Concept MRI.
Loads test data and runs the app for interactive testing.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_mri.app import app
from concept_mri.tests.test_data_generators import TestDataGenerator


def load_test_data_into_app():
    """Load test data into app stores for manual testing."""
    print("Generating test data...")
    
    # Generate complete test state
    test_state = TestDataGenerator.generate_complete_test_state(
        num_layers=12,
        include_ets=True
    )
    
    # Convert numpy arrays to lists for JSON compatibility
    def convert_arrays(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        return obj
    
    test_state = convert_arrays(test_state)
    
    print("Test data generated successfully!")
    print("\nTest Configuration:")
    print(f"- Model: 12 layers, 768 dimensions, 100 samples")
    print(f"- Clustering: ETS with hierarchy support")
    print(f"- Windows: GPT-2 style (Early, Middle, Late)")
    print(f"- Paths: 50 top paths analyzed")
    print(f"- Labels: LLM-generated cluster labels")
    
    # Note: In a real scenario, you would load this data through the UI
    # For testing, you can manually set the stores using browser developer tools
    
    # Save test data for loading
    test_file = Path(__file__).parent / 'manual_test_data.json'
    with open(test_file, 'w') as f:
        json.dump(test_state, f, indent=2)
    
    print(f"\nTest data saved to: {test_file}")
    print("\nTo load test data in the app:")
    print("1. Open browser developer console")
    print("2. Run: localStorage.setItem('model-store', JSON.stringify(testData['model-store']))")
    print("3. Run: sessionStorage.setItem('clustering-store', JSON.stringify(testData['clustering-store']))")
    print("4. Refresh the page")
    
    return test_state


def main():
    """Run the app with test data."""
    print("""
    ╔════════════════════════════════════════╗
    ║     CONCEPT MRI - MANUAL TEST MODE     ║
    ╚════════════════════════════════════════╝
    """)
    
    # Generate test data
    test_data = load_test_data_into_app()
    
    print("\nStarting Concept MRI in test mode...")
    print("Navigate to http://localhost:8050")
    print("\nTest Scenarios to Try:")
    print("1. Change hierarchy level (Macro/Meso/Micro) and observe cluster count changes")
    print("2. Apply different window presets and see how visualizations update")
    print("3. Switch between clustering algorithms (K-Means/DBSCAN/ETS)")
    print("4. Try different visualization modes in Stepped Trajectory")
    print("5. Explore cluster cards with different display options")
    print("6. Test export functionality for visualizations")
    
    # Run the app
    app.run_server(debug=True, host='localhost', port=8050)


if __name__ == '__main__':
    main()