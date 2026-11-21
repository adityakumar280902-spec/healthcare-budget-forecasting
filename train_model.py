import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 1. Load dataset
df = pd.read_csv("hospital_operations.csv", parse_dates=["date"])

# Add day_of_year so the model captures the smooth seasonal wave
df["day_of_year"] = df["date"].dt.dayofyear

features = ["department", "day_of_year", "month", "day_of_week", "is_weekend"]
targets = ["admissions", "nurses_needed", "doctors_needed", "total_budget"]

# 2. Chronological split: train on 2024, test on 2025
train_df = df[df["date"].dt.year == 2024]
test_df = df[df["date"].dt.year == 2025]

X_train, y_train = train_df[features], train_df[targets]
X_test, y_test = test_df[features], test_df[targets]

# 3. Preprocessing + Model Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("dept", OneHotEncoder(handle_unknown="ignore"), ["department"]),
        ("num", "passthrough", ["day_of_year", "month", "day_of_week", "is_weekend"]),
    ]
)

base_rf = RandomForestRegressor(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", MultiOutputRegressor(base_rf)),
    ]
)

# 4. Train on 2024 data
pipeline.fit(X_train, y_train)

# 5. Evaluate on unseen 2025 data
preds = pipeline.predict(X_test)

print("=== 2025 Test Set Evaluation ===")
for i, col in enumerate(targets):
    mae = mean_absolute_error(y_test[col], preds[:, i])
    r2 = r2_score(y_test[col], preds[:, i])
    if col == "total_budget":
        print(f"{col:<15} | MAE: INR {mae:,.0f} | R2: {r2:.3f}")
    else:
        print(f"{col:<15} | MAE: {mae:.2f} | R2: {r2:.3f}")

# 6. Sanity check with a sample future date (Saturday in peak October season)
sample = pd.DataFrame([
    {
        "department": "Emergency",
        "day_of_year": 288,
        "month": 10,
        "day_of_week": 5,
        "is_weekend": 1,
    }
])
sample_pred = pipeline.predict(sample)[0]

print("\n=== Sample Forecast: Emergency (Oct Weekend) ===")
print(f"Predicted Admissions : {round(sample_pred[0])}")
print(f"Nurses Needed        : {round(sample_pred[1])}")
print(f"Doctors Needed       : {round(sample_pred[2])}")
print(f"Estimated Budget     : INR {sample_pred[3]:,.0f}")

# 7. Save the trained pipeline
joblib.dump(pipeline, "model.pkl")
print("\nSaved trained pipeline to model.pkl")