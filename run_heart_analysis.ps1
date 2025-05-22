# run_heart_analysis.ps1
# Script to run just the heart dataset analysis to test our fixes

# Clean up previous files for heart dataset
Remove-Item -Path "results\cluster_paths\heart*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data\cluster_paths\heart*.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "visualization\data\cluster_paths\heart*.json" -Force -ErrorAction SilentlyContinue

# Run cluster paths analysis for heart with use_full_dataset flag
python run_analysis.py cluster_paths --dataset heart --seed 0 --compute_similarity --config_id baseline --max_k 10 --output_dir data/cluster_paths --use_full_dataset

# Run cluster stats analysis for heart
python run_analysis.py cluster_stats --dataset heart --seed 0

# Run similarity metrics analysis for heart
python run_analysis.py similarity_metrics --dataset heart --seed 0 --similarity_metric cosine

# Run cross-layer metrics analysis for heart
python run_analysis.py cross_layer_metrics --dataset heart --seed 0

Write-Host "Heart dataset analysis completed!" -ForegroundColor Green