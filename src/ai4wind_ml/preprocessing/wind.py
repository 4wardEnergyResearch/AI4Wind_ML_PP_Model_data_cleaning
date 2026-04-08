# ai4wind_ml/preprocessing/wind.py
"""Load wind speed data from ERA5 NetCDF files for a turbine location."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from ..config import TurbineConfig


class WindDataPreprocessor:
    """Loads wind speed data from ERA5 NetCDF files
    and prepares it as feature input for the ML model.

    Spatial extraction uses a KD-tree on the lat/lon grid to find the
    nearest grid cell to the turbine location.  Results are cached so the
    tree is built only once.
    """

    def __init__(
        self,
        config: TurbineConfig,
        netcdf_dir: str | Path = Path("data/wind_speed_data/historical_data"),
        variable: str = "ensemble",
    ):
        self.config = config
        self.netcdf_dir = Path(netcdf_dir)
        self.variable = variable

        # Cached spatial index
        self._tree: cKDTree | None = None
        self._lat2d: np.ndarray | None = None
        self._iy: int | None = None
        self._ix: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_era5_wind_speed(self, start: str, end: str) -> pd.Series:
        """Extract an hourly wind speed time series for the turbine location.

        Parameters
        ----------
        start, end : str
            ISO date strings (e.g. ``"2020-01-01"``, ``"2020-12-31"``).

        Returns
        -------
        pd.Series
            Hourly wind speed (m/s) with a datetime index.
        """
        height = self._nearest_available_height()
        nc_files = self._resolve_nc_files(start, end, height)
        if not nc_files:
            raise FileNotFoundError(
                f"No NetCDF files found in {self.netcdf_dir} for height "
                f"{height}m between {start} and {end}"
            )

        # Build spatial index from the first file (cached)
        self._ensure_spatial_index(nc_files[0])

        # Extract time series from each file and concatenate
        series_parts = []
        for nc_file in nc_files:
            s = self._extract_point(nc_file)
            series_parts.append(s)

        ts = pd.concat(series_parts).sort_index()
        ts = ts[~ts.index.duplicated(keep="first")]
        ts = ts.loc[start:end]
        ts.name = "wind_speed"
        return ts

    def get_features(self, start: str, end: str) -> pd.DataFrame:
        """Return wind speed as a single-column feature DataFrame."""
        speed = self.load_era5_wind_speed(start, end)
        return pd.DataFrame({"wind_speed": speed})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nearest_available_height(self) -> int:
        """Find the available height closest to the turbine hub height."""
        available = self._detect_heights()
        if not available:
            raise FileNotFoundError(
                f"No NetCDF files found in {self.netcdf_dir}"
            )
        target = self.config.hub_height_m
        return min(available, key=lambda h: abs(h - target))

    def _detect_heights(self) -> list[int]:
        """Scan the NetCDF directory for available height levels."""
        heights = set()
        for f in self.netcdf_dir.glob("*.nc"):
            m = re.search(r"_(\d+)m_", f.name)
            if m:
                heights.add(int(m.group(1)))
        return sorted(heights)

    def _resolve_nc_files(
        self, start: str, end: str, height: int
    ) -> list[Path]:
        """Find monthly NetCDF files covering [start, end] for *height*."""
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        files = sorted(
            f
            for f in self.netcdf_dir.glob(f"*_{height}m_*.nc")
        )

        # Filter to files whose month falls within the requested range
        selected = []
        for f in files:
            m = re.search(r"(\d{4})-(\d{2})\.nc$", f.name)
            if m:
                file_date = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-01")
                file_end = file_date + pd.offsets.MonthEnd(1)
                if file_end >= start_dt and file_date <= end_dt:
                    selected.append(f)

        return selected

    def _ensure_spatial_index(self, sample_nc: Path) -> None:
        """Build the KD-tree from the first NetCDF file (once)."""
        if self._tree is not None:
            return

        ds = xr.open_dataset(sample_nc, engine="netcdf4")
        try:
            self._lat2d = ds["lat"].values
            lon2d = ds["lon"].values
        finally:
            ds.close()

        # Convert lat/lon to Cartesian for KD-tree
        phi = np.radians(self._lat2d)
        lam = np.radians(self._wrap_lon(lon2d))
        xyz = np.column_stack((
            np.cos(phi.flatten()) * np.cos(lam.flatten()),
            np.cos(phi.flatten()) * np.sin(lam.flatten()),
            np.sin(phi.flatten()),
        ))
        self._tree = cKDTree(xyz)

        # Query for the turbine location
        self._iy, self._ix = self._nearest_indices(
            self.config.latitude, self.config.longitude
        )

    def _nearest_indices(self, lat: float, lon: float) -> tuple[int, int]:
        """Find the (iy, ix) grid indices nearest to (lat, lon)."""
        phi = np.radians(lat)
        lam = np.radians(self._wrap_lon(lon))
        point = [np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)]
        _, idx = self._tree.query(point, k=1)
        iy, ix = np.unravel_index(idx, self._lat2d.shape)
        return int(iy), int(ix)

    def _extract_point(self, nc_file: Path) -> pd.Series:
        """Extract the time series at the cached (iy, ix) from one file."""
        ds = xr.open_dataset(nc_file, engine="netcdf4")
        try:
            arr = ds[self.variable].isel(y=self._iy, x=self._ix).load()
            s = arr.to_pandas()
            s.index = pd.to_datetime(s.index)
        finally:
            ds.close()
        return s

    @staticmethod
    def _wrap_lon(lon: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, dtype=np.float64)
        return (lon + 180) % 360 - 180
