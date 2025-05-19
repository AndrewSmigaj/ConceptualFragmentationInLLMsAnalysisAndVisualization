@echo off
echo Running Cluster Paths generation for Titanic dataset with seed 0...
venv311\Scripts\python.exe -m concept_fragmentation.analysis.cluster_paths --compute_similarity --dataset titanic --seed 0

echo.
echo Running Dashboard...
venv311\Scripts\python.exe -m visualization.dash_app

echo.
echo Done!
pause