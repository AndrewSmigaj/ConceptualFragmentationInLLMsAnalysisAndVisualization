# Baseline Experiments Runner Script for Concept Fragmentation Project
# This script runs the baseline experiments for all datasets with three seeds

# Ensure Python environment is activated
# Assuming you have a virtual environment in the project root

# Change this if you want to use specific seeds
$seeds = 0, 1, 2

# Define datasets to run
$datasets = "titanic", "adult", "heart", "fashion_mnist"

# Set up log file
$logDir = ".\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "$logDir\baseline_run_$timestamp.log"

# Function to log messages
function Log-Message {
    param (
        [string]$message
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $message" | Out-File -FilePath $logFile -Append
    Write-Host "$timestamp - $message"
}

Log-Message "Starting baseline experiments"
Log-Message "Using seeds: $seeds"
Log-Message "Using datasets: $datasets"

# Check if GPU is available
$gpuAvailable = python -c "import torch; print(torch.cuda.is_available())" | Out-String
$device = if ($gpuAvailable.Trim() -eq "True") { "cuda" } else { "cpu" }
Log-Message "Using device: $device"

# Run experiments for each dataset
foreach ($dataset in $datasets) {
    Log-Message "Running experiments for dataset: $dataset"
    
    # Run for each seed
    foreach ($seed in $seeds) {
        Log-Message "Running experiment for dataset: $dataset with seed: $seed"
        
        # Run the experiment
        try {
            python -m concept_fragmentation.experiments.baseline_run --datasets $dataset --seeds $seed --device $device
            Log-Message "Successfully completed experiment for $dataset with seed $seed"
        }
        catch {
            Log-Message "ERROR: Failed to run experiment for $dataset with seed $seed - $_"
        }
    }
}

# Generate baseline summary
Log-Message "Generating baseline summary for all experiments"
python -m concept_fragmentation.experiments.baseline_run

Log-Message "All baseline experiments completed. See results in ./results/baselines/" 