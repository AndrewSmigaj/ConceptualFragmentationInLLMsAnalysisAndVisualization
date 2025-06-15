"""
Run this script to diagnose and fix path issues.
"""

import os
import sys

# Get the current directory
current_dir = os.path.abspath(os.getcwd())
print(f"Current directory: {current_dir}")

# Add the current directory to Python's path
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"Added {current_dir} to Python path")

# Try importing the module
try:
    import concept_fragmentation
    print(f"Successfully imported concept_fragmentation from {concept_fragmentation.__file__}")
except ImportError as e:
    print(f"Error importing concept_fragmentation: {e}")
    
# Print the full Python path
print("\nPython path:")
for path in sys.path:
    print(f"  {path}")

print("\nTo fix path issues, run this command in PowerShell:")
print(f"$env:PYTHONPATH = '{current_dir}'")
print("\nOr in CMD:")
print(f"set PYTHONPATH={current_dir}")