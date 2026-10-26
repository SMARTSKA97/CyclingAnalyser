## 1. Initialise Venv(if not done)
python -m venv .venv

## 2. Activate Venv
.\.venv\Scripts\Activate.ps1

or

.venv\Scripts\activate.bat

## 3. Install dependencies (if not done)
python -m pip install streamlit pandas numpy plotly openpyxl

## 4. Run the application
streamlit run cycling_workout_analyzer.py