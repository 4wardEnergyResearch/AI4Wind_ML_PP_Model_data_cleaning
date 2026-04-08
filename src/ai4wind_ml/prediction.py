# ai4wind_ml/prediction.py
"""Load a trained model and generate power predictions."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TurbineConfig
from .model import WindTurbineModel


class PowerPredictor:
    """Loads a trained model for a turbine and generates power predictions.

    Parameters
    ----------
    config : TurbineConfig
        Turbine configuration.
    model_dir : Path, optional
        Directory containing saved ``.pkl`` models.
        Defaults to ``data/ml_models/``.
    """

    def __init__(
        self,
        config: TurbineConfig,
        model_dir: str | Path = Path("data/ml_models"),
    ):
        self.config = config
        self.model_dir = Path(model_dir)
        self._model: WindTurbineModel | None = None

    def load_model(self) -> None:
        """Load the model from disk."""
        self._model = WindTurbineModel.load(
            self.model_dir, self.config.turbine_id
        )

    def predict(self, wind_features: pd.DataFrame) -> pd.Series:
        """Generate power time series from a wind feature DataFrame."""
        if self._model is None:
            self.load_model()
        return pd.Series(
            self._model.predict(wind_features),
            index=wind_features.index,
            name=f"power_{self.config.turbine_id}",
        )
