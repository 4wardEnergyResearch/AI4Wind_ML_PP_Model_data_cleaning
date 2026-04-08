# ai4wind_ml/training.py
"""Model training with auto-selection and direct-fit modes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import TurbineConfig
from .model import WindTurbineModel


# ======================================================================
# Default pipeline registry
# ======================================================================

def default_pipelines() -> dict[str, Pipeline]:
    """Return the set of candidate pipelines with default hyperparameters."""
    return {
        "XGBoost": Pipeline([
            ("Scaler", StandardScaler()),
            ("XGB", XGBRegressor()),
        ]),
        "GradientBoosting": Pipeline([
            ("Scaler", StandardScaler()),
            ("GBR", GradientBoostingRegressor()),
        ]),
        "RandomForest": Pipeline([
            ("Scaler", StandardScaler()),
            ("RF", RandomForestRegressor()),
        ]),
        "MLP": Pipeline([
            ("Scaler", StandardScaler()),
            ("MLP", MLPRegressor(max_iter=500)),
        ]),
        "Ridge": Pipeline([
            ("Scaler", StandardScaler()),
            ("Ridge", Ridge()),
        ]),
    }


def default_param_grids() -> dict[str, dict]:
    """Return the default hyperparameter grids for GridSearchCV."""
    return {
        "XGBoost": {
            "XGB__n_estimators": [100, 200],
            "XGB__max_depth": [1, 2, 3, 4, 5],
        },
        "GradientBoosting": {
            "GBR__n_estimators": [100, 200, 300],
            "GBR__max_depth": [1, 2, 3, 4, 5, 6],
        },
        "RandomForest": {
            "RF__n_estimators": [100, 200, 300],
            "RF__max_depth": [1, 2, 3, 4, 5, 6],
        },
        "MLP": {
            "MLP__hidden_layer_sizes": [(100,), (100, 100)],
            "MLP__learning_rate_init": [0.01, 0.1],
        },
        "Ridge": {
            "Ridge__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    }


# ======================================================================
# Metrics helper
# ======================================================================

def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute RMSE, MAE, and R² for predictions."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ======================================================================
# ModelTrainer
# ======================================================================

class ModelTrainer:
    """Train an ML model for a turbine with two modes:

    **Auto mode** (default): tests all candidate pipelines via
    GridSearchCV and selects the best one by R² score.

    **Direct mode**: fits a single user-supplied pipeline directly
    without cross-validation.

    Parameters
    ----------
    config : TurbineConfig
        Turbine configuration.
    model_dir : Path, optional
        Where to save trained models.  Defaults to ``data/ml_models/``.
    """

    def __init__(
        self,
        config: TurbineConfig,
        model_dir: str | Path = Path("data/ml_models"),
    ):
        self.config = config
        self.model_dir = Path(model_dir)

    # ------------------------------------------------------------------
    # Auto mode
    # ------------------------------------------------------------------

    def fit_auto(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        pipelines: dict[str, Pipeline] | None = None,
        param_grids: dict[str, dict] | None = None,
        cv: int = 3,
        scoring: str = "r2",
        test_size: float = 0.3,
        random_state: int | None = None,
        verbose: int = 1,
    ) -> WindTurbineModel:
        """Test all candidate pipelines via GridSearchCV, pick the best,
        evaluate on a held-out test set, and save the model.

        Parameters
        ----------
        X : DataFrame
            Feature matrix (e.g. columns ``wind_speed``, optionally
            ``wind_direction``).
        y : Series
            Target power values.
        pipelines : dict, optional
            ``{name: sklearn.Pipeline}``.  Defaults to :func:`default_pipelines`.
        param_grids : dict, optional
            ``{name: {param: values}}``.  Defaults to :func:`default_param_grids`.
        cv : int
            Cross-validation folds.
        scoring : str
            Scoring metric for GridSearchCV.
        test_size : float
            Fraction held out for final evaluation.
        random_state : int, optional
            Random seed for reproducibility.
        verbose : int
            Verbosity level for GridSearchCV.

        Returns
        -------
        WindTurbineModel
            The best model, already saved to disk.
        """
        if pipelines is None:
            pipelines = default_pipelines()
        if param_grids is None:
            param_grids = default_param_grids()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # --- Run GridSearchCV for each pipeline ---
        results: list[dict] = []
        searches: list[GridSearchCV] = []

        for name, pipe in pipelines.items():
            grid = param_grids.get(name, {})
            print(f"Optimising {name} ...")
            search = GridSearchCV(
                pipe, grid, cv=cv, scoring=scoring, verbose=verbose,
                refit=True,
            )
            search.fit(X_train, y_train)
            searches.append(search)

            y_pred = search.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            results.append({
                "name": name,
                "best_cv_score": search.best_score_,
                "best_params": search.best_params_,
                "test_RMSE": metrics["RMSE"],
                "test_MAE": metrics["MAE"],
                "test_R2": metrics["R2"],
                "refit_time": search.refit_time_,
            })
            print(
                f"  {name}: CV R²={search.best_score_:.4f}, "
                f"test R²={metrics['R2']:.4f}"
            )

        # --- Select best by test R² ---
        comparison = pd.DataFrame(results)
        best_idx = int(comparison["test_R2"].argmax())
        best_name = comparison.loc[best_idx, "name"]
        best_search = searches[best_idx]
        best_pipeline = best_search.best_estimator_

        print(f"\nBest model: {best_name} (test R²={comparison.loc[best_idx, 'test_R2']:.4f})")

        # --- Wrap and save ---
        model = WindTurbineModel(
            turbine_id=self.config.turbine_id,
            pipeline=best_pipeline,
            feature_names=list(X.columns),
            metadata={
                "trained_on": str(pd.Timestamp.now()),
                "mode": "auto",
                "best_model": best_name,
                "best_params": best_search.best_params_,
                "cv_score": best_search.best_score_,
                "test_metrics": results[best_idx],
                "comparison": comparison.to_dict(orient="records"),
            },
        )
        model.save(self.model_dir)
        return model

    # ------------------------------------------------------------------
    # Direct mode
    # ------------------------------------------------------------------

    def fit_direct(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        pipeline: Pipeline,
    ) -> WindTurbineModel:
        """Fit a single pipeline directly without cross-validation.

        Parameters
        ----------
        X : DataFrame
            Feature matrix.
        y : Series
            Target power values.
        pipeline : Pipeline
            A fully configured sklearn Pipeline to fit.

        Returns
        -------
        WindTurbineModel
            The fitted model, saved to disk.
        """
        pipeline.fit(X, y)
        model = WindTurbineModel(
            turbine_id=self.config.turbine_id,
            pipeline=pipeline,
            feature_names=list(X.columns),
            metadata={
                "trained_on": str(pd.Timestamp.now()),
                "mode": "direct",
            },
        )
        model.save(self.model_dir)
        return model
