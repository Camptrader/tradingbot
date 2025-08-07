@echo off
REM Change directory to the folder where this script is saved
cd /d %~dp0

REM Activate the virtual environment (make sure .venv exists in your project)
call .venv\Scripts\activate

REM Launch the Streamlit app
streamlit run app.py

REM Keep the window open to show any errors
pause
