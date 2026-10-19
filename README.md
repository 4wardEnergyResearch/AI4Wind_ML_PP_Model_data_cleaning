<a id="readme-top"></a>
<!--
Adapted from the Best-README-Template available at
https://github.com/othneildrew/Best-README-Template/
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GPL v3 License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://www.4wardenergy.at/de/referenzen/ai4wind">
    <img src="https://www.4wardenergy.at/fileadmin/user_upload/AI4Wind_Logo_400x400-01.jpg" alt="Logo" width="300" height="300">
  </a>

<h3 align="center">AI4WIND ML: Wind Turbine Power Prediction Pipeline</h3>

  <p align="center">
    A modular Python package for cleaning wind turbine SCADA data, training ML models, and predicting power output from gridded wind speed data.
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-this-repository">About This Repository</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#folder-structure">Folder Structure</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#data-requirements">Data Requirements</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT -->
## About This Repository

This repository contains a standalone ML pipeline for wind turbine power prediction, developed in the course of the [AI4Wind](https://www.4wardenergy.at/de/referenzen/ai4wind) project. It provides an end-to-end workflow from raw turbine measurement data to trained power prediction models, designed for climate impact assessment of Austrian wind farms.

The pipeline was developed and validated on operational SCADA data from **315 wind turbines** across Austria, spanning 30 turbine types from multiple manufacturers with hub heights from 31 to 166 m and rated powers from 225 kW to 7.6 MW. Wind speed input features are extracted from the INCA/ERA5 gridded analysis product provided by [GeoSphere Austria](https://www.geosphere.at), available at multiple height levels (80--220 m) with hourly resolution and approximately 1 km spatial resolution.

The package includes:
- A **four-stage data cleaning routine** for quality-flagging turbine measurement data
- **Automated ML model selection** comparing five regression algorithms via cross-validated grid search
- **ERA5 wind speed extraction** from NetCDF files using spatial KD-tree lookup

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- FEATURES -->
## Features

- **Four-stage data cleaning** with configurable parameters per turbine type:
  1. Invalid data flagging (out-of-range, NaN, operational inconsistencies)
  2. Power curve shift envelope detection
  3. Sigma clipping with Savitzky-Golay smoothing
  4. Quantile-based lower envelope flagging
- **ERA5/INCA wind speed extraction** from NetCDF grids at the nearest grid point and height level to each turbine
- **Auto model selection**: compares Gradient Boosting, XGBoost, Random Forest, MLP, and Ridge regression via GridSearchCV
- **Direct-fit mode**: train a specific pipeline with chosen hyperparameters, without cross-validation
- **Turbine configuration from database**: load metadata, power curves, and cleaning parameters from an SQLite database
- **Publication-ready src layout** installable via `pip` or `uv`

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- FOLDER STRUCTURE -->
## Folder Structure

```
src/ai4wind_ml/                  Python package (src layout)
├── config.py                    TurbineConfig and CleaningConfig dataclasses
├── model.py                     WindTurbineModel (sklearn pipeline wrapper)
├── training.py                  ModelTrainer with auto and direct modes
├── prediction.py                PowerPredictor
├── pipeline.py                  WindTurbinePipeline (high-level orchestrator)
└── preprocessing/
    ├── wind.py                  ERA5 NetCDF wind speed extraction
    └── power.py                 SCADA data cleaning (4-stage flagging)

data/
├── wind_speed_data/
│   ├── historical_data/         ERA5 NetCDF files for training (hourly, 2020)
│   └── future_scenarios/        Climate projection NetCDF files
├── turbine_data/
│   ├── datenbank.db             SQLite database (turbine metadata + power curves)
│   ├── raw/                     Raw 10-min SCADA measurement CSVs
│   └── cleaned/
│       ├── csv/                 Flagged time series output
│       └── diagnostics/         Per-turbine cleaning diagnostic plots
└── ml_models/                   Saved model artifacts (.pkl)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

- **Python 3.11** or later
- [**uv**](https://docs.astral.sh/uv/) for dependency management (recommended), or pip

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning.git
cd AI4Wind_ML_PP_Model_data_cleaning
uv sync
```

This installs the `ai4wind_ml` package in development mode along with all dependencies.

### Verify installation

```bash
uv run python example.py
```

If the pipeline runs through without errors, the installation is complete.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE -->
## Usage

The full workflow is demonstrated in [`example.py`](example.py). Here is a condensed overview:

### 1. Load turbine configuration

```python
from ai4wind_ml import TurbineConfig

config = TurbineConfig.from_database("data/turbine_data/datenbank.db", "2214")
```

This loads turbine metadata (coordinates, hub height, rated power), the manufacturer power curve, and optional cleaning parameters from the SQLite database.

### 2. Clean raw SCADA data

```python
from ai4wind_ml import PowerDataPreprocessor

pp = PowerDataPreprocessor(config)
raw = pp.load_raw_data()
power_clean, flagged_data = pp.clean(raw)
pp.export_cleaning_report()
```

Applies the four-stage flagging pipeline and exports a diagnostic scatter plot.

When processing multiple turbines, use `export_fleet_cleaning_report()` to generate a summary Excel workbook:

```python
from ai4wind_ml import export_fleet_cleaning_report

export_fleet_cleaning_report(results)
```

This produces a formatted xlsx with flag counts and ratios per turbine, color-scaled columns, clickable links to diagnostic plots, and automatic highlighting of suspicious turbines (>10% flagged).

### 3. Extract ERA5 wind speed

```python
from ai4wind_ml import WindDataPreprocessor

wp = WindDataPreprocessor(config)
X = wp.get_features("2020-01-01", "2020-12-31")
```

Extracts hourly wind speed at the nearest grid point and height level from ERA5 NetCDF files.

### 4. Train ML model

**Auto mode** (compare all candidates, pick the best):

```python
from ai4wind_ml import ModelTrainer

trainer = ModelTrainer(config)
model = trainer.fit_auto(X_train, y_train)
```

**Direct mode** (fit a specific pipeline):

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

pipe = Pipeline([("Scaler", StandardScaler()), ("XGB", XGBRegressor(n_estimators=200, max_depth=3))])
model = trainer.fit_direct(X_train, y_train, pipe)
```

### 5. Predict

```python
from ai4wind_ml import PowerPredictor

predictor = PowerPredictor(config)
power = predictor.predict(X)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- DATA REQUIREMENTS -->
## Data Requirements

### Turbine data

Raw SCADA CSV files in `data/turbine_data/raw/` with the following columns:

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 datetime |
| `power [kW]` | Turbine power output |
| `windspeed [m/s]` | Hub-height wind speed |
| `temperature [C]` | Ambient temperature (optional) |
| `wind direction [deg]` | Wind direction (optional) |
| `gondel position [deg]` | Nacelle position (optional) |

Turbine metadata and manufacturer power curves are stored in an SQLite database (`datenbank.db`). See the provided database for the expected schema.

### Cleaning configuration

The data cleaning routine can be tuned per turbine type via `data/turbine_data/data_cleaning_config.xlsx`. Each row corresponds to a turbine type and controls the behaviour of stages 2--4:

| Parameter | Description |
|---|---|
| `perform_pc_shift` | Enable/disable power curve shift flagging |
| `pc_shift_ws` | Wind speed shift (% of rated wind speed) |
| `pc_shift_pw` | Power shift (% of nominal power) |
| `cut_thresh_pc` | Wind speed threshold above which PC shift is not applied |
| `perform_sigma_clipping` | Enable/disable sigma clipping |
| `sc_mult` | Sigma multiplier (number of standard deviations) |
| `cut_thresh_sc` | Wind speed threshold above which sigma clipping is not applied |
| `perform_quantile_mapping` | Enable/disable quantile-based flagging |
| `num_quantiles` | Number of quantile divisions |
| `thresh_fall` | Threshold for detecting power drops in quantile derivative |
| `cut_thresh_qm` | Wind speed threshold above which quantile mapping is not applied |

The provided config file is pre-populated with tuned parameters for 30 common Austrian turbine types. If no config file is present, default parameters are used. The configuration is loaded automatically when using `TurbineConfig.from_database()` if the xlsx file is located next to the database.

### Wind speed data

Gridded ERA5/INCA wind speed fields in NetCDF format, placed in `data/wind_speed_data/historical_data/`. Files follow the naming convention:

```
INCA_ERA5_2020_wind_{HEIGHT}m_2020-{MM}.nc
```

Available from GeoSphere Austria. The package uses the `ensemble` variable and selects the height level nearest to the turbine's hub height.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html.en)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

[Project website](https://www.4wardenergy.at/de/referenzen/ai4wind)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning.svg?style=for-the-badge
[contributors-url]: https://github.com/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning.svg?style=for-the-badge
[forks-url]: https://github.com/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning/network/members
[stars-shield]: https://img.shields.io/github/stars/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning.svg?style=for-the-badge
[stars-url]: https://github.com/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning/stargazers
[issues-shield]: https://img.shields.io/github/issues/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning.svg?style=for-the-badge
[issues-url]: https://github.com/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning/issues
[license-shield]: https://img.shields.io/github/license/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning.svg?style=for-the-badge
[license-url]: https://github.com/4wardEnergyResearch/AI4Wind_ML_PP_Model_data_cleaning/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/4ward-energy-research-gmbh/
