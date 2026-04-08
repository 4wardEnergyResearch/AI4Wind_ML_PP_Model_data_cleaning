"""
AI4WIND ML Pipeline Example
============================

Demonstrates the full pipeline on a single turbine:
1. Load turbine configuration from the database
2. Clean raw SCADA data (4-stage flagging)
3. Extract ERA5 wind speed features
4. Train ML models (auto mode: compare all candidates)
5. Generate predictions from the trained model
"""

from ai4wind_ml import (
    TurbineConfig,
    PowerDataPreprocessor,
    WindDataPreprocessor,
    ModelTrainer,
    WindTurbineModel,
    PowerPredictor,
)

TURBINE_ID = "2214"
DB_PATH = "data/turbine_data/datenbank.db"
WIND_START = "2020-01-01"
WIND_END = "2020-12-31"

# ── 1. Load turbine configuration ──────────────────────────────────────

config = TurbineConfig.from_database(DB_PATH, TURBINE_ID)
print(f"Turbine {config.turbine_id} ({config.turbine_type})")
print(f"  Location: {config.latitude:.4f}N, {config.longitude:.4f}E")
print(f"  Hub height: {config.hub_height_m} m, Rated power: {config.rated_power_kw} kW\n")

# ── 2. Clean raw SCADA data ────────────────────────────────────────────

pp = PowerDataPreprocessor(config)
raw = pp.load_raw_data()
print(f"Raw data: {len(raw):,} rows ({raw.index[0]} to {raw.index[-1]})")

power_clean, flagged = pp.clean(raw)
print(f"Cleaning results:")
for flag, count in flagged["Flag Computed"].value_counts().sort_index().items():
    pct = count / len(flagged) * 100
    print(f"  Flag {flag}: {count:>8,} ({pct:.1f}%)")

pp.export_cleaning_report()
print(f"Diagnostic plot saved to data/turbine_data/cleaned/diagnostics/\n")

# ── 3. Extract ERA5 wind speed features ────────────────────────────────

wp = WindDataPreprocessor(config)
X = wp.get_features(WIND_START, WIND_END)
print(f"ERA5 wind speed: {len(X):,} hourly values")
print(f"  Height used: {wp._nearest_available_height()} m")
print(f"  Range: {X['wind_speed'].min():.1f} -- {X['wind_speed'].max():.1f} m/s\n")

# ── 4. Align and train ─────────────────────────────────────────────────

y = power_clean.resample("1h").mean()
common = X.index.intersection(y.dropna().index)
X_train, y_train = X.loc[common], y.loc[common]
print(f"Training data: {len(X_train):,} aligned hourly samples\n")

trainer = ModelTrainer(config, model_dir="data/ml_models")
model = trainer.fit_auto(X_train, y_train, verbose=0)
print(f"\nBest model: {model.metadata['best_model']}")
print(f"  Test R^2: {model.metadata['test_metrics']['test_R2']:.3f}")
print(f"  Test RMSE: {model.metadata['test_metrics']['test_RMSE']:.1f} kW")
print(f"  Best params: {model.metadata['best_params']}")

print("\nAll candidates:")
for r in model.metadata["comparison"]:
    print(f"  {r['name']:20s}  CV R2={r['best_cv_score']:.3f}  Test R2={r['test_R2']:.3f}")

# ── 5. Load saved model and predict ────────────────────────────────────

loaded = WindTurbineModel.load("data/ml_models", TURBINE_ID)
predictor = PowerPredictor(config, model_dir="data/ml_models")
power_pred = predictor.predict(X)

print(f"\nPredictions: {len(power_pred):,} hourly values")
print(f"  Range: {power_pred.min():.1f} -- {power_pred.max():.1f} kW")
print(f"  Mean:  {power_pred.mean():.1f} kW")
