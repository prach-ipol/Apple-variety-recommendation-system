@echo off
echo Starting Recommendation System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit
)

REM Install required packages
echo Installing required packages...
python -m pip install flask pandas numpy scikit-learn openpyxl

REM Run the application
echo.
echo Starting the application...
echo Please open http://localhost:5000 in your web browser
echo.
python app.py

pause 