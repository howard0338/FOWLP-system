@echo off
cd /d "%~dp0"

echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed: python not found. Install Python 3.10+ and add to PATH.
    exit /b 1
)

echo [2/3] Activating venv...
call venv\Scripts\activate.bat

echo [3/3] Installing packages...
pip install --upgrade pip
pip install streamlit numpy pandas plotly scipy

echo.
echo Done. Run the simulator with:
echo   call venv\Scripts\activate.bat
echo   streamlit run app.py
pause
