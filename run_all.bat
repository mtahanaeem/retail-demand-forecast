@echo off
TITLE Pipeline Launcher

cd /d "%~dp0"

echo Step 1/5: Installing Python dependencies...
cd "C:\Users\Tahan\Desktop\Test\project2-time-series-forecasting\sales-demand-forecasting"
pip install -r requirements.txt
echo.
pause

echo Step 2/5: Installing frontend dependencies...
cd "C:\Users\Tahan\Desktop\Test\project2-time-series-forecasting\sales-demand-forecasting\frontend"
call npm install
echo.
pause

echo Step 3/5: Starting backend...
cd "C:\Users\Tahan\Desktop\Test\project2-time-series-forecasting\sales-demand-forecasting"
start "FastAPI-Backend" cmd /k "uvicorn api.main:app --reload --host 0.0.0.0 --port 8000"
echo Backend starting in separate window...
echo.
pause

echo Step 4/5: Starting frontend...
cd "C:\Users\Tahan\Desktop\Test\project2-time-series-forecasting\sales-demand-forecasting\frontend"
start "React-Frontend" cmd /k "npm run dev"
echo Frontend starting in separate window...
echo.
pause

echo Step 5/5: Done!
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
