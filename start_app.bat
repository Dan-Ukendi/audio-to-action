@echo off
rem Double-click to open the local app in your browser (http://localhost:8501). Close this window to stop it.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m streamlit run app\app.py
