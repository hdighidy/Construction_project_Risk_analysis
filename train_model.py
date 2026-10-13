"""Train a baseline model on synthetic data. Replace with governed real project data before operational use."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score

DATA = Path("data")
MODEL = Path("models")
MODEL.mkdir(exist_ok=True)

NUMERIC = [
    "budget_egp", "planned_duration_months", "complexity_1_5", "design_maturity_pct",
    "procurement_risk_1_5", "subcontractor_risk_1_5", "site_readiness_pct",
    "change_order_rate_pct", "payment_delay_days", "resource_gap_pct",
    "approval_delay_days", "quality_ncr_rate", "safety_incidents", "long_lead_material_months"
]
CATEGORICAL = ["project_type", "discipline"]
FEATURES = NUMERIC + CATEGORICAL

def main():
    df = pd.read_csv(DATA/"mep_projects.csv")
    # Training target is the synthetic delay probability proxy, not a validated outcome.
    # For real training, use actual final outcomes and prediction-time-safe features.
    train = df.dropna(subset=["delay_days_final"]).copy()
    X = train[FEATURES]
    y_delay = train["delay_days_final"].astype(float)
    y_overrun = train["cost_overrun_pct_final"].astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y_delay, test_size=.2, random_state=42)
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    delay_model = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1))])
    delay_model.fit(Xtr, ytr)
    pred = delay_model.predict(Xte)
    print(f"Delay-days MAE (synthetic holdout): {mean_absolute_error(yte, pred):.2f} days")
    delay_model.fit(X, y_delay)
    joblib.dump(delay_model, MODEL/"delay_days_model.joblib")

    Xtr, Xte, ytr, yte = train_test_split(X, y_overrun, test_size=.2, random_state=42)
    overrun_model = Pipeline([("prep", prep), ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=43, n_jobs=-1))])
    overrun_model.fit(Xtr, ytr)
    print(f"Cost-overrun MAE (synthetic holdout): {mean_absolute_error(yte, overrun_model.predict(Xte)):.2f} percentage points")
    overrun_model.fit(X, y_overrun)
    joblib.dump(overrun_model, MODEL/"cost_overrun_model.joblib")
    print("Saved models in models/. Synthetic metrics are not evidence of real-world performance.")

if __name__ == "__main__":
    main()
