# Housekeeping script to clean up legacy folders and retrain heart dataset

Write-Host "Step 1: Removing legacy heart folders..." -ForegroundColor Cyan
# Remove legacy Heart folders that sit directly under RESULTS_DIR
Remove-Item -Path "D:\concept_fragmentation_results\heart_baseline_seed*" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Step 2: Retraining Heart baseline model..." -ForegroundColor Cyan
# Retrain Heart baseline (activations saved by default)
python -m concept_fragmentation.experiments.train --dataset heart --seeds 0

Write-Host "Step 3: Running full analysis..." -ForegroundColor Cyan
# Run full analysis - now finds the new folder & new filenames
.\clean-and-run-analysis.ps1

Write-Host "Verification Checklist:" -ForegroundColor Green
Write-Host "• Check for new Heart folder path:" -ForegroundColor Yellow
Write-Host "  D:\concept_fragmentation_results\baselines\heart\heart_baseline_seed0_<timestamp>" -ForegroundColor White
Write-Host "• Files now produced:" -ForegroundColor Yellow
Write-Host "  data/cluster_paths/heart_seed_0_paths.json" -ForegroundColor White
Write-Host "  data/cluster_paths/heart_seed_0_paths_with_centroids.json" -ForegroundColor White
Write-Host "• Dataset Summary should show non-zero counts and 'Seed 0' for Heart & Titanic" -ForegroundColor Yellow