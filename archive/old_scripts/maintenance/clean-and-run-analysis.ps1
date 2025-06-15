# clean-and-run-analysis.ps1

# Clean up previous files
Remove-Item -Path "results\cluster_paths\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "data\cluster_paths\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "results\cluster_stats\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "data\cluster_stats\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "results\similarity_metrics\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "data\similarity_metrics\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "results\cross_layer_metrics\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "data\cross_layer_metrics\*" -Force -Recurse -ErrorAction SilentlyContinue
Remove-Item -Path "analysis_results.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data\silhouette_scores.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data\dataset_info.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "results\silhouette_table.md" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "results\silhouette_table.tex" -Force -ErrorAction SilentlyContinue

# 1. Generate cluster paths for titanic
python run_analysis.py cluster_paths --dataset titanic --seed 0 --compute_similarity --config_id baseline --max_k 10 --output_dir data/cluster_paths

# 2. Generate cluster paths for heart - using special flag to ensure full dataset is used
python run_analysis.py cluster_paths --dataset heart --seed 0 --compute_similarity --config_id baseline --max_k 10 --output_dir data/cluster_paths --use_full_dataset

# 3. Generate cluster statistics for titanic
python run_analysis.py cluster_stats --dataset titanic --seed 0

# 4. Generate cluster statistics for heart
python run_analysis.py cluster_stats --dataset heart --seed 0

# 5. Compute similarity metrics for titanic
python run_analysis.py similarity_metrics --dataset titanic --seed 0 --similarity_metric cosine

# 6. Compute similarity metrics for heart
python run_analysis.py similarity_metrics --dataset heart --seed 0 --similarity_metric cosine

# 7. Compute cross-layer metrics for titanic
python run_analysis.py cross_layer_metrics --dataset titanic --seed 0

# 8. Compute cross-layer metrics for heart
python run_analysis.py cross_layer_metrics --dataset heart --seed 0

# 9. Compile dataset info
python visualization\improved_compile_dataset_info.py

# 10. Extract silhouette scores
python visualization\basic_extract_silhouette_scores.py

Write-Host "All analysis scripts completed successfully!" -ForegroundColor Green