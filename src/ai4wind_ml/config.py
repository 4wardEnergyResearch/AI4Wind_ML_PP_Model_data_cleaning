# ai4wind_ml/config.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class CleaningConfig:
    """Per-turbine-type parameters for the multi-stage flagging pipeline.

    Can be loaded from the legacy data_cleaning_config.xlsx via ``from_excel()``.
    """
    # Power curve shift
    perform_pc_shift: bool = True
    pc_shift_ws: float = 15.0        # % of rated wind speed
    pc_shift_pw: float = 10.0        # % of nominal power
    cut_thresh_pc: float = float("nan")

    # Sigma clipping
    perform_sigma_clipping: bool = True
    sc_mult: float = 3.0
    cut_thresh_sc: float = float("nan")

    # Quantile-based envelope
    perform_quantile_mapping: bool = True
    num_quantiles: int = 100
    thresh_fall: float = -1.0
    cut_thresh_qm: float = float("nan")

    @classmethod
    def from_excel(cls, path: str | Path, turbine_type: str) -> CleaningConfig:
        """Load cleaning parameters for *turbine_type* from an xlsx config file."""
        df = pd.read_excel(path)
        row = df[df["turbine_type"] == turbine_type]
        if row.empty:
            raise ValueError(
                f"Turbine type '{turbine_type}' not found in {path}. "
                f"Available types: {df['turbine_type'].tolist()}"
            )
        r = row.iloc[0]
        return cls(
            perform_pc_shift=bool(r["perform_pc_shift"]),
            pc_shift_ws=float(r["pc_shift_ws"]),
            pc_shift_pw=float(r["pc_shift_pw"]),
            cut_thresh_pc=float(r["cut_thresh_pc"]),
            perform_sigma_clipping=bool(r["perform_sigma_clipping"]),
            sc_mult=float(r["sc_mult"]),
            cut_thresh_sc=float(r["cut_thresh_sc"]),
            perform_quantile_mapping=bool(r["perform_quantile_mapping"]),
            num_quantiles=int(r["num_quantiles"]),
            thresh_fall=float(r["thresh_fall"]),
            cut_thresh_qm=float(r["cut_thresh_qm"]),
        )


@dataclass
class TurbineConfig:
    """Configuration for a single wind turbine."""
    turbine_id: str
    latitude: float
    longitude: float
    hub_height_m: float
    rated_power_kw: float

    turbine_type: str | None = None
    power_curve: pd.DataFrame | None = field(default=None, repr=False)
    cleaning_config: CleaningConfig | None = None

    @classmethod
    def from_database(cls, db_path: str | Path, turbine_id: str) -> TurbineConfig:
        """Create a TurbineConfig by querying the legacy SQLite database.

        Populates turbine metadata, power curve, and derives cleaning config
        if a ``data_cleaning_config.xlsx`` is found next to the database.
        """
        db_path = Path(db_path)
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT [turbine type], [nominal power], [hub height], "
                "[latitude], [longitude] "
                "FROM metadaten WHERE [turbine ID] = ?",
                (turbine_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Turbine '{turbine_id}' not found in database")

            turbine_type, nominal_power, hub_height, lat, lon = row

            # Load power curve from per-type table (first two columns only)
            cur.execute(f"SELECT windspeed, power FROM '{turbine_type}'")
            pc_rows = cur.fetchall()
            power_curve = pd.DataFrame(pc_rows, columns=["windspeed", "power"])
        finally:
            conn.close()

        # Try to load cleaning config from xlsx next to database
        cleaning_config = None
        config_xlsx = db_path.parent / "data_cleaning_config.xlsx"
        if config_xlsx.exists():
            try:
                cleaning_config = CleaningConfig.from_excel(config_xlsx, turbine_type)
            except ValueError:
                pass  # turbine type not in config — use defaults

        return cls(
            turbine_id=turbine_id,
            latitude=lat,
            longitude=lon,
            hub_height_m=hub_height,
            rated_power_kw=nominal_power,
            turbine_type=turbine_type,
            power_curve=power_curve,
            cleaning_config=cleaning_config,
        )
