@echo off
echo ================================
echo OMR Evaluation System Setup
echo ================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8+
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ================================
echo Setup complete!
echo ================================
echo.
echo To start the application:
echo 1. Run: venv\Scripts\activate.bat
echo 2. Run: python app.py
echo.
echo Then open: http://localhost:5000
pause
