@echo off
REM One-time setup: virtual environment + packages
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo Setup complete. Next: run_pipeline.bat, then run_app.bat
