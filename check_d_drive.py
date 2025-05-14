import os
import sys

# Set the results directory to D drive
results_dir = "D:/concept_fragmentation_results"

print(f"Checking access to directory: {results_dir}")

try:
    # Try to create the directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    print(f"✓ Successfully created directory: {results_dir}")
    
    # Try to create a test file
    test_file = os.path.join(results_dir, "test_file.txt")
    with open(test_file, "w") as f:
        f.write("Test file to check write permissions")
    print(f"✓ Successfully wrote test file: {test_file}")
    
    # Create baseline directory
    baselines_dir = os.path.join(results_dir, "baselines")
    os.makedirs(baselines_dir, exist_ok=True)
    print(f"✓ Successfully created baselines directory: {baselines_dir}")
    
    print("\nD drive is accessible and ready to use for experiment results.")
    print("All experiments will now be stored in D:/concept_fragmentation_results")
    
except Exception as e:
    print(f"❌ ERROR: Could not access or write to D drive: {str(e)}")
    print("\nPlease make sure:")
    print("1. The D drive exists on your system")
    print("2. You have write permissions to the D drive")
    print("3. There is enough space on the D drive")
    sys.exit(1) 