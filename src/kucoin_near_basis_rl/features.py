from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


FEATURE_COLUMNS = [
    "basis",
    "basis_zscore",
    "spot_volatility",
    "futures_volatility",
    "volume_imbalance",
    "basis_momentum",
]


def build_feature_frame(raw_frame: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    frame = raw_frame.copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    frame["basis"] = (frame["futures_close"] - frame["spot_close"]) / frame["spot_close"]
    frame["spot_return"] = np.log(frame["spot_close"]).diff()
    frame["futures_return"] = np.log(frame["futures_close"]).diff()

    basis_mean = frame["basis"].rolling(cfg.zscore_window).mean()
    basis_std = frame["basis"].rolling(cfg.zscore_window).std()
    frame["basis_zscore"] = (frame["basis"] - basis_mean) / basis_std.replace(0.0, np.nan)

    frame["spot_volatility"] = frame["spot_return"].rolling(cfg.volatility_window).std()
    frame["futures_volatility"] = frame["futures_return"].rolling(cfg.volatility_window).std()

    spot_volume_ma = frame["spot_volume"].rolling(cfg.volume_window).mean()
    futures_volume_ma = frame["futures_volume"].rolling(cfg.volume_window).mean()
    frame["volume_imbalance"] = (futures_volume_ma - spot_volume_ma) / (
        futures_volume_ma + spot_volume_ma + 1e-12
    )

    frame["basis_momentum"] = frame["basis"].diff(cfg.basis_momentum_lag)

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return frame
