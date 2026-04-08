# ai4wind_ml/model.py
"""Wrapper around a trained sklearn Pipeline with metadata."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline


@dataclass
class WindTurbineModel:
    """Stores a trained sklearn Pipeline alongside turbine and training metadata.

    Attributes
    ----------
    turbine_id : str
        Identifier of the turbine this model was trained for.
    pipeline : Pipeline
        The fitted sklearn Pipeline.
    feature_names : list[str]
        Column names expected by the pipeline.
    metadata : dict
        Training metadata (date, mode, CV scores, best params, etc.).
    """
    turbine_id: str
    pipeline: Pipeline
    feature_names: list[str]
    metadata: dict = field(default_factory=dict)

    def predict(self, X) -> np.ndarray:
        """Generate predictions from the fitted pipeline."""
        return self.pipeline.predict(X)

    def save(self, model_dir: str | Path) -> Path:
        """Persist the model to ``{model_dir}/{turbine_id}.pkl``."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{self.turbine_id}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, model_dir: str | Path, turbine_id: str) -> WindTurbineModel:
        """Load a previously saved model."""
        path = Path(model_dir) / f"{turbine_id}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
