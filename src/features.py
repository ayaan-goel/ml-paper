"""Seasonal weather features computed from daily NASA POWER arrays.

All functions are NaN-aware so the same code builds features from complete
daily data and from daily data with simulated outages (block missingness).
"""
import warnings

import numpy as np

# Order of variables in the daily arrays
VARS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"]
RAIN, T2M, TMAX, TMIN, RH, SRAD, WIND = range(7)

# Season windows: (start month, start day, length in days); a Rabi window starts in
# November of the first year of the agricultural year and ends in March of the next
SEASONS = {"Kharif": (6, 1, 153), "Rabi": (11, 1, 151)}
HOT_THRESHOLD = {"Kharif": 35.0, "Rabi": 32.0}  # deg C, Tmax for a "hot day"
WET_DAY_MM = 2.5                                # IMD rainy-day threshold
MAX_MISSING = 0.5                               # feature is NaN if more of its days are missing

FEATURE_GROUPS = {
    "rain": ["rain_total", "rain_days", "rain_sd", "rain_max1d", "rain_early", "rain_mid", "rain_late"],
    "temp": ["tmean", "tmax_mean", "tmin_mean", "hot_days", "tmean_early", "tmean_mid", "tmean_late"],
    "humidity": ["rh_mean"],
    "solar": ["srad_mean"],
    "wind": ["wind_mean"],
}
WEATHER_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def _guard(values, x, axis=1):
    """Set a feature to NaN where more than MAX_MISSING of its days are missing."""
    frac_missing = np.isnan(x).mean(axis=axis)
    out = values.astype(float).copy()
    out[frac_missing > MAX_MISSING] = np.nan
    return out


def _total(x):
    """Seasonal total, rescaled to the full window when some days are missing."""
    with np.errstate(invalid="ignore"):
        return _guard(np.nanmean(x, axis=1) * x.shape[1], x)


def _mean(x):
    with np.errstate(invalid="ignore"):
        return _guard(np.nanmean(x, axis=1), x)


def _count(x, cond):
    """Count of days meeting cond, rescaled to the full window."""
    obs = ~np.isnan(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (cond & obs).sum(axis=1) / obs.sum(axis=1)
    return _guard(frac * x.shape[1], x)


def _thirds(x):
    n = x.shape[1]
    a, b = n // 3, 2 * n // 3
    return x[:, :a], x[:, a:b], x[:, b:]


def seasonal_features(daily, season):
    """daily: array (rows, days, 7) for one season window. Returns dict of 1-D arrays."""
    rain, tmean, tmax = daily[:, :, RAIN], daily[:, :, T2M], daily[:, :, TMAX]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = {
            "rain_total": _total(rain),
            "rain_days": _count(rain, np.nan_to_num(rain) >= WET_DAY_MM),
            "rain_sd": _guard(np.nanstd(rain, axis=1), rain),
            "rain_max1d": _guard(np.nanmax(np.where(np.isnan(rain), -np.inf, rain), axis=1), rain),
            "tmean": _mean(tmean),
            "tmax_mean": _mean(tmax),
            "tmin_mean": _mean(daily[:, :, TMIN]),
            "hot_days": _count(tmax, np.nan_to_num(tmax) > HOT_THRESHOLD[season]),
            "rh_mean": _mean(daily[:, :, RH]),
            "srad_mean": _mean(daily[:, :, SRAD]),
            "wind_mean": _mean(daily[:, :, WIND]),
        }
        for name, part in zip(["early", "mid", "late"], _thirds(rain)):
            f[f"rain_{name}"] = _total(part)
        for name, part in zip(["early", "mid", "late"], _thirds(tmean)):
            f[f"tmean_{name}"] = _mean(part)
    return {k: f[k] for k in WEATHER_FEATURES}
