# Healthcare Resource Allocation & Budget Forecasting

An AI-driven forecasting system that predicts daily hospital patient admissions, nursing and physician staffing needs, and operational budgets across departments (**Emergency, ICU, General Ward, and Pediatrics**).

- **Problem Domain:** AI in Healthcare and Finance
- **UN SDG Alignment:** SDG 3 (Good Health and Well-being), SDG 12 (Responsible Consumption & Production)

## Key Features
- **Seasonal & Day-of-Week Forecasting:** Captures monsoon/winter patient surges and weekend Emergency spikes.
- **Multi-Output ML Pipeline:** Single `scikit-learn` pipeline (`OneHotEncoder` + `MultiOutputRegressor(RandomForestRegressor)`) trained chronologically on 2024 data and validated on unseen 2025 data.
- **Interactive Planning Dashboard:** Streamlit web app with 1–90 day forecast horizons, department filters, an outbreak/inflation surge buffer slider (0–30%), and one-click CSV export.

## Project Structure
- `generate_data.py` — Generates a 2-year daily hospital operations dataset (`hospital_operations.csv`)
- `train_model.py` — Trains, evaluates (MAE & R² on 2025 holdout), and exports `model.pkl`
- `app.py` — Interactive Streamlit dashboard for administrators
- `requirements.txt` — Python dependencies

## Run Locally
```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
streamlit run app.py
