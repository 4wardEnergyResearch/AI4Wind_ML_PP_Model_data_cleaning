# ai4wind_ml/pipeline.py
"""High-level orchestrator for wind turbine power prediction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from .config import TurbineConfig
from .model import WindTurbineModel
from .preprocessing.power import PowerDataPreprocessor
from .preprocessing.wind import WindDataPreprocessor
from .prediction import PowerPredictor
from .training import ModelTrainer


class WindTurbinePipeline:
    """High-level orchestrator.  Apply to any turbine via :class:`TurbineConfig`.

    Example — auto mode (test all models, pick best)
    -------------------------------------------------
    >>> config = TurbineConfig.from_database("data/turbine_data/datenbank.db", "2214")
    >>> wtp = WindTurbinePipeline(config)
    >>> wtp.train(start="2020-01-01", end="2020-12-31")

    Example — direct mode (fit a specific pipeline)
    ------------------------------------------------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from xgboost import XGBRegressor
    >>> pipe = Pipeline([("Scaler", StandardScaler()), ("XGB", XGBRegressor(n_estimators=200, max_depth=3))])
    >>> wtp.train(start="2020-01-01", end="2020-12-31", pipeline=pipe)
    """

    def __init__(
        self,
        config: TurbineConfig,
        netcdf_dir: str | Path = Path("data/wind_speed_data/historical_data"),
        model_dir: str | Path = Path("data/ml_models"),
    ):
        self.config = config
        self.model_dir = Path(model_dir)
        self.wind_prep = WindDataPreprocessor(config, netcdf_dir=netcdf_dir)
        self.power_prep = PowerDataPreprocessor(config)
        self.trainer = ModelTrainer(config, model_dir=self.model_dir)
        self.predictor = PowerPredictor(config, model_dir=self.model_dir)

    def train(
        self,
        start: str,
        end: str,
        *,
        freq: str = "1h",
        pipeline: Pipeline | None = None,
        **auto_kwargs,
    ) -> WindTurbineModel:
        """Train a model using wind features and cleaned power data.

        If *pipeline* is provided, uses **direct mode**: fits that
        specific pipeline without cross-validation.

        Otherwise, uses **auto mode**: tests all default candidate
        pipelines via GridSearchCV and picks the best one.  Extra
        keyword arguments are forwarded to
        :meth:`ModelTrainer.fit_auto`.

        Parameters
        ----------
        start, end : str
            Date range for wind feature extraction (ISO format).
        freq : str
            Resampling frequency to align wind (hourly) with power
            (10-min).  Default ``"1h"``.
        pipeline : Pipeline, optional
            If given, fit this pipeline directly (no CV).
        **auto_kwargs
            Forwarded to :meth:`ModelTrainer.fit_auto` (e.g.
            ``cv``, ``test_size``, ``verbose``, ``pipelines``,
            ``param_grids``).
        """
        X, y = self._prepare_training_data(start, end, freq)

        if pipeline is not None:
            return self.trainer.fit_direct(X, y, pipeline)
        else:
            return self.trainer.fit_auto(X, y, **auto_kwargs)

    def predict(self, start: str, end: str) -> pd.Series:
        """Generate power predictions from wind features."""
        X = self.wind_prep.get_features(start, end)
        return self.predictor.predict(X)

    def _prepare_training_data(
        self, start: str, end: str, freq: str = "1h"
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load and align wind features with cleaned power target."""
        X = self.wind_prep.get_features(start, end)
        y_raw = self.power_prep.get_target()

        # Aggregate 10-min power to match wind frequency
        y = y_raw.resample(freq).mean()

        # Align on common timestamps
        common = X.index.intersection(y.dropna().index)
        return X.loc[common], y.loc[common]
