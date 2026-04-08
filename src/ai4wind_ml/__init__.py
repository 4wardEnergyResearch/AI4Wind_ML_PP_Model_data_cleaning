# ai4wind_ml/__init__.py
"""
AI4WIND ML - Wind Turbine Power Prediction Package

A modular machine learning package for wind turbine power prediction
using ERA5 wind speed data and turbine measurements.
"""

from .config import CleaningConfig, TurbineConfig
from .pipeline import WindTurbinePipeline
from .model import WindTurbineModel
from .training import ModelTrainer, default_pipelines, default_param_grids
from .prediction import PowerPredictor
from .preprocessing.wind import WindDataPreprocessor
from .preprocessing.power import PowerDataPreprocessor, export_fleet_cleaning_report

__version__ = "0.1.0"

__all__ = [
    "CleaningConfig",
    "TurbineConfig",
    "WindTurbinePipeline",
    "WindTurbineModel",
    "ModelTrainer",
    "default_pipelines",
    "default_param_grids",
    "PowerPredictor",
    "WindDataPreprocessor",
    "PowerDataPreprocessor",
    "export_fleet_cleaning_report",
]
