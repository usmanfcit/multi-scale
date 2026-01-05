@echo off
echo Starting Backend Server...
echo.
echo Make sure you have:
echo 1. Python 3.11 installed
echo 2. Virtual environment activated (.venv)
echo 3. Dependencies installed (pip install .)
echo 4. .env file configured
echo.
echo Starting server on http://localhost:8000
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause

