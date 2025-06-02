"""
Run Concept MRI app with proper path configuration.
"""
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the app
from concept_mri.app import app

if __name__ == "__main__":
    print("Starting Concept MRI app...")
    print("Navigate to http://localhost:8050 in your browser")
    app.run_server(debug=True, host='0.0.0.0', port=8050)