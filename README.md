# Robust Crop Yield Prediction Under Missing Weather Data and Geographic Distribution Shift

Code and data pipeline for a study of how district-level crop-yield models for Gujarat, India
behave when weather inputs are incomplete and when they are applied to districts not seen
during training.

## Data

| Source | Content |
|---|---|
| Directorate of Economics and Statistics (DES), Ministry of Agriculture and Farmers Welfare | District-wise, season-wise crop area, production and yield, 1997-98 to 2022-23 |
| NASA POWER (daily point API) | Precipitation, temperature (mean, max, min), relative humidity, solar radiation and wind speed, 1997-2023 |

Study scope: all 33 districts of Gujarat; groundnut and bajra (Kharif) and wheat (Rabi).

## Reproducing the results

```bash
pip install -r requirements.txt
python src/download_crop_data.py     # DES crop statistics (about 56 MB)
python src/download_nasa_power.py    # daily weather for 33 district headquarters
python src/prepare_data.py           # builds data/processed/panel.csv
python src/experiments.py            # runs experiments E1-E4
```

## Repository layout

```
src/
  download_crop_data.py   crop statistics download
  download_nasa_power.py  weather download
  features.py             seasonal weather features (NaN-aware)
  prepare_data.py         cleaning, joining and data audit
  experiments.py          experiments E1-E4
data/processed/           analysis dataset
results/                  predictions, tables and figures
```

## Experiments

- **E1** Complete data under random, temporal and district-holdout evaluation
- **E2** Simulated missing weather (random, variable-wise, block outages; 10-40%) with median,
  KNN, iterative and native (XGBoost) handling, in familiar and unseen districts
- **E3** Feature selection (mutual information, recursive feature elimination)
- **E4** Location features (latitude, longitude)
