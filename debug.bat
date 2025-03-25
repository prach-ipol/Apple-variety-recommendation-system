@echo off
echo Starting Recommendation System in Debug Mode...
echo.

REM Check Python installation
echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit
)

REM Check required packages
echo.
echo Checking required packages...
python -m pip list | findstr "flask pandas numpy scikit-learn openpyxl"
if errorlevel 1 (
    echo Installing required packages...
    python -m pip install flask pandas numpy scikit-learn openpyxl
)

REM Check if dataset exists
echo.
echo Checking dataset file...
if exist dataset1.xlsx (
    echo Found dataset1.xlsx
) else (
    echo WARNING: dataset1.xlsx not found
    echo Using sample data instead
)

REM Run the application
echo.
echo Starting the application...
echo Please open http://localhost:5000 in your web browser
echo.
python app.py

pause 