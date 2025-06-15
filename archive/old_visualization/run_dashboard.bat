@echo off
echo Starting Neural Network Trajectory Explorer Dashboard...
echo.

rem Activate the virtual environment (if available)
if exist venv311\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv311\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python.
)

echo.
echo Running dashboard...
echo.

python visualization\run_dashboard.py

if %ERRORLEVEL% NEQ 0 (
    echo Error running dashboard. Please ensure all dependencies are installed.
    echo Install dependencies with: pip install -r visualization\requirements.txt
    pause
)

echo.
echo Done!
echo.
pause