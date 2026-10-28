"""Build the analysis dataset: Gujarat district x crop x season x year.

Inputs  data/raw/crop/crop-wise-area-production-yield.csv  (DES APY statistics)
        data/raw/nasa_power/<District>.csv                 (daily NASA POWER)
Outputs data/processed/panel.csv          one row per observation, with complete features
        data/processed/daily_<crop>.npz   daily season-window arrays for each row
        results/tables/data_audit.csv     records kept at each cleaning step
"""
import json, pathlib, sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from features import SEASONS, VARS, WEATHER_FEATURES, seasonal_features

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_CROP = ROOT / "data/raw/crop/crop-wise-area-production-yield.csv"
RAW_WX = ROOT / "data/raw/nasa_power"
OUT = ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)
(ROOT / "results/tables").mkdir(parents=True, exist_ok=True)

CROPS = {"Groundnut": "Kharif", "Bajra": "Kharif", "Wheat": "Rabi"}
MIN_AREA_HA = 1000  # records below this are dropped: tiny areas give unreliable yields


def read_power_csv(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("-END HEADER-") + 1
    df = pd.read_csv(path, skiprows=start)
    df["date"] = pd.to_datetime(df["YEAR"].astype(str), format="%Y") + pd.to_timedelta(df["DOY"] - 1, unit="D")
    df = df.set_index("date")[VARS].astype(float)
    return df.replace(-999.0, np.nan)


def main():
    audit = []
    crop = pd.read_csv(RAW_CROP)
    audit.append(("All India records", len(crop)))
    g = crop[crop.state_name == "Gujarat"]
    audit.append(("Gujarat records", len(g)))
    keep = np.zeros(len(g), bool)
    for c, s in CROPS.items():
        keep |= ((g.crop_name == c) & (g.season == s)).to_numpy()
    g = g[keep].copy()
    audit.append(("Groundnut/Bajra (Kharif), Wheat (Rabi)", len(g)))
    g = g.dropna(subset=["area", "production"])
    g = g[(g.area > 0) & (g.production > 0)]
    audit.append(("Valid area and production", len(g)))
    g = g[g.area >= MIN_AREA_HA]
    audit.append((f"Area >= {MIN_AREA_HA} ha", len(g)))
    assert not g.duplicated(["year", "district_name", "crop_name"]).any()

    g["year_start"] = g.year.str[:4].astype(int)
    g["yield_t_ha"] = g.production / g.area

    coords = {}
    for line in (RAW_WX / "_download_log.json").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        coords[r["district"]] = (r["lat"], r["lon"])
    missing = sorted(set(g.district_name) - set(coords))
    assert not missing, f"No weather for districts: {missing}"
    wx = {d: read_power_csv(RAW_WX / f"{d}.csv") for d in coords}

    panel_parts = []
    for c, season in CROPS.items():
        sub = g[g.crop_name == c].sort_values(["district_name", "year_start"]).reset_index(drop=True)
        month, day, length = SEASONS[season]
        daily = np.empty((len(sub), length, len(VARS)))
        for i, r in enumerate(sub.itertuples()):
            start = pd.Timestamp(r.year_start, month, day)
            block = wx[r.district_name].loc[start: start + pd.Timedelta(days=length - 1)]
            assert len(block) == length, (r.district_name, r.year_start)
            daily[i] = block.to_numpy()
        assert not np.isnan(daily).any(), "Unexpected missing values in raw weather"
        feats = pd.DataFrame(seasonal_features(daily, season))
        lat = sub.district_name.map(lambda d: coords[d][0])
        lon = sub.district_name.map(lambda d: coords[d][1])
        part = pd.concat([sub[["district_name", "crop_name", "season", "year", "year_start",
                                "area", "production", "yield_t_ha"]].reset_index(drop=True),
                          pd.DataFrame({"lat": lat, "lon": lon}), feats], axis=1)
        part.insert(0, "row_id", np.arange(len(part)))
        np.savez_compressed(OUT / f"daily_{c.lower()}.npz", daily=daily, row_id=part.row_id.to_numpy())
        panel_parts.append(part)
        audit.append((f"{c}: observations with weather", len(part)))

    panel = pd.concat(panel_parts, ignore_index=True)
    panel.to_csv(OUT / "panel.csv", index=False)
    pd.DataFrame(audit, columns=["step", "records"]).to_csv(ROOT / "results/tables/data_audit.csv", index=False)
    for step, n in audit:
        print(f"{step:45s} {n:>8,}")
    print(panel.groupby("crop_name").agg(rows=("yield_t_ha", "size"), districts=("district_name", "nunique"),
                                         first=("year_start", "min"), last=("year_start", "max"),
                                         mean_yield=("yield_t_ha", "mean")).round(3))


if __name__ == "__main__":
    main()
