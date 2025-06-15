#!/usr/bin/env pwsh
# generate_critical_metrics.ps1
# Script to generate all critical analysis files for Titanic and Heart datasets

Write-Host "===== Generating Critical Analysis Files =====" -ForegroundColor Cyan

# 1. Generate cluster statistics files
Write-Host "`nGenerating cluster statistics..." -ForegroundColor Yellow
python -m concept_fragmentation.analysis.cluster_stats --dataset titanic --seed 0 --output_format json
python -m concept_fragmentation.analysis.cluster_stats --dataset heart --seed 0 --output_format json

# 2. Generate silhouette scores
Write-Host "`nGenerating silhouette scores..." -ForegroundColor Yellow
python -m concept_fragmentation.analysis.silhouette_scores --dataset titanic --seed 0
python -m concept_fragmentation.analysis.silhouette_scores --dataset heart --seed 0

# 3. Generate cross-layer metrics with explicit output paths
Write-Host "`nGenerating cross-layer metrics..." -ForegroundColor Yellow
python -m concept_fragmentation.analysis.cross_layer_metrics --dataset titanic --seed 0 --output_dir data/cross_layer
python -m concept_fragmentation.analysis.cross_layer_metrics --dataset heart --seed 0 --output_dir data/cross_layer

# 4. Generate similarity matrices
Write-Host "`nGenerating similarity matrices..." -ForegroundColor Yellow
python -m concept_fragmentation.analysis.similarity_metrics --dataset titanic --seed 0 --similarity_metric cosine --output_format json
python -m concept_fragmentation.analysis.similarity_metrics --dataset heart --seed 0 --similarity_metric cosine --output_format json

# 5. Run the comprehensive metrics pipeline
Write-Host "`nRunning comprehensive metrics analysis..." -ForegroundColor Yellow
python -m concept_fragmentation.analysis.run_all_metrics --dataset titanic --seed 0 
python -m concept_fragmentation.analysis.run_all_metrics --dataset heart --seed 0

# 6. Update dataset summary
Write-Host "`nUpdating dataset summary..." -ForegroundColor Yellow
python .\visualization\improved_compile_dataset_info.py

Write-Host "`n===== Analysis Complete =====" -ForegroundColor Green
Write-Host "All critical analysis files have been generated." -ForegroundColor Green 