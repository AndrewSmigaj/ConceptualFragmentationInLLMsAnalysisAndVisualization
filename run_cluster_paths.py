"""
Helper script to run the cluster_paths.py with proper Python path setup.
"""

import os
import sys
import subprocess

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Get command line arguments
    import sys
    args = sys.argv[1:]
    
    # Construct the command
    cmd = [sys.executable, 'concept_fragmentation/analysis/cluster_paths.py'] + args
    
    # Run the script with the correct path
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root
    
    try:
        subprocess.run(cmd, env=env, check=True)
        print("Cluster paths script completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running cluster paths script: {e}")