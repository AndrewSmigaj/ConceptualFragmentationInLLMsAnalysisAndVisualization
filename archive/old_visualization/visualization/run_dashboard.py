"""
Simple script to run the dashboard.

This script ensures the dashboard runs with the proper Python environment.
"""

import os
import sys

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the dashboard
from visualization.dash_app import app

if __name__ == "__main__":
    print("Starting Neural Network Trajectory Explorer dashboard...")
    print("Navigate to http://127.0.0.1:8050/ in your web browser.")
    app.run_server(debug=True, port=8050)