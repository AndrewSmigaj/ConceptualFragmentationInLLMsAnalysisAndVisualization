# run_full_pipeline.ps1
# Comprehensive script to run the full analysis pipeline

Write-Host "Starting full pipeline..." -ForegroundColor Cyan

# We rely on concept_fragmentation/llm/api_keys.py for credentials – no env-var prompt needed

# Step 1: Clean previous results and generate cluster paths
Write-Host "`nStep 1: Generating cluster paths and statistics..." -ForegroundColor Green
.\clean-and-run-analysis.ps1

# Step 1.5: Generate proper cluster statistics
Write-Host "`nStep 1.5: Generating real cluster statistics..." -ForegroundColor Green
# Make sure the stats directory exists
if (-not (Test-Path -Path "data/cluster_stats")) {
    New-Item -ItemType Directory -Path "data/cluster_stats" -Force
}
# Generate statistics for titanic
python tools/run_cluster_stats.py --dataset titanic --seed 0
# Generate statistics for heart
python tools/run_cluster_stats.py --dataset heart --seed 0

# Step 2: Run LLM path analysis for Titanic dataset
Write-Host "`nStep 2a: Running LLM analysis for Titanic dataset (Holistic Mode)..." -ForegroundColor Green
python llm_path_analysis.py --dataset titanic --seed 0 --output_dir results/llm --outfile titanic_seed_0_analysis.json --analysis_mode holistic --num_paths 7

# Step 3: Run LLM path analysis for Heart dataset
Write-Host "`nStep 2b: Running LLM analysis for Heart dataset (Holistic Mode)..." -ForegroundColor Green
python llm_path_analysis.py --dataset heart --seed 0 --output_dir results/llm --outfile heart_seed_0_analysis.json --analysis_mode holistic --num_paths 7

# Step 4: Build paper tables and content
Write-Host "`nStep 3: Building paper tables and content..." -ForegroundColor Green
python tools/build_paper_tables.py

# Step 5: Generate paper figures
Write-Host "`nStep 4: Generating paper figures..." -ForegroundColor Green
python generate_paper_figures.py

# Step 6: Integrate figures into paper
Write-Host "`nStep 5: Integrating figures into paper..." -ForegroundColor Green
python integrate_figures.py

# Step 7: Build the paper (if latex is available)
Write-Host "`nStep 6: Building paper..." -ForegroundColor Green
$latex_dir = "arxiv_submission"
$current_dir = Get-Location
Set-Location $latex_dir
$latex_available = Get-Command pdflatex -ErrorAction SilentlyContinue

if ($latex_available) {
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex
    Write-Host "Paper built successfully. PDF available at: $latex_dir\main.pdf" -ForegroundColor Green
} else {
    Write-Host "LaTeX not available. Skipping paper build." -ForegroundColor Yellow
}
Set-Location $current_dir

Write-Host "`nFull pipeline completed!" -ForegroundColor Cyan 