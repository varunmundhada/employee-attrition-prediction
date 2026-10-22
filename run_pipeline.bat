@echo off
REM EDA charts, then train + compare models and save the best one
.venv\Scripts\python -m src.eda || exit /b 1
.venv\Scripts\python -m src.train || exit /b 1
echo Pipeline finished. Start the app with run_app.bat
