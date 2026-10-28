"""Run all experiments and save out-of-fold predictions.

E1  complete data under three evaluation designs (random, temporal, district holdout)
E2  simulated missing weather (MCAR, variable-wise, block) x imputation, under all three
    designs (random split = familiar districts and years; temporal; district holdout)
E3  feature selection (mutual information, RFE) under the district holdout
E4  location features (latitude, longitude) under the district holdout

Usage: python src/experiments.py [E1 E2 E3 E4]
Outputs: results/predictions_<exp>.csv.gz (one row per observation and condition)
"""
import itertools, pathlib, sys, time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from features import FEATURE_GROUPS, SEASONS, WEATHER_FEATURES, seasonal_features

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed"
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]
N_FOLDS = 5
TEMPORAL_CUTOFF = 2015          # train on agricultural years starting <= 2015, test on later
RATES = [0.1, 0.2, 0.3, 0.4]
PATTERNS = ["mcar", "variable", "block"]
BASE_FEATURES = WEATHER_FEATURES + ["year_start"]
N_SELECT = 8                    # size of the reduced feature sets in E3


# ---------------------------------------------------------------- models
def make_model(name, seed):
    if name == "Mean":
        return DummyRegressor(strategy="mean")
    if name == "Ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == "RF":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5,
                                     random_state=seed, n_jobs=1)
    if name == "XGB":
        return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=3, random_state=seed, n_jobs=1)
    raise ValueError(name)


def make_imputer(name, seed):
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "knn":
        return make_pipeline(StandardScaler(), KNNImputer(n_neighbors=5))
    if name == "iterative":
        return make_pipeline(StandardScaler(), IterativeImputer(max_iter=10, random_state=seed))
    raise ValueError(name)


def fit_predict(model, imputer, Xtr, ytr, Xte, seed):
    steps = []
    if imputer not in (None, "native"):
        steps.append(make_imputer(imputer, seed))
    steps.append(make_model(model, seed))
    pipe = make_pipeline(*steps)
    pipe.fit(Xtr, ytr)
    return pipe.predict(Xte)


# ---------------------------------------------------------------- splits
def splits(design, df, seed):
    """Yield (train_idx, test_idx) for a design."""
    idx = np.arange(len(df))
    if design == "random":
        yield from KFold(N_FOLDS, shuffle=True, random_state=seed).split(idx)
    elif design == "temporal":
        tr = idx[df.year_start.to_numpy() <= TEMPORAL_CUTOFF]
        te = idx[df.year_start.to_numpy() > TEMPORAL_CUTOFF]
        yield tr, te
    elif design == "district":
        gkf = GroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        yield from gkf.split(idx, groups=df.district_name.to_numpy())
    else:
        raise ValueError(design)


# ---------------------------------------------------------------- missingness
def apply_missingness(df, daily, season, pattern, rate, seed):
    """Return a copy of the feature matrix (BASE_FEATURES) with simulated missing weather."""
    rng = np.random.default_rng(seed)
    X = df[BASE_FEATURES].to_numpy(dtype=float).copy()
    nw = len(WEATHER_FEATURES)
    if pattern == "mcar":
        mask = rng.random((len(df), nw)) < rate
        X[:, :nw][mask] = np.nan
    elif pattern == "variable":
        for feats in FEATURE_GROUPS.values():
            cols = [WEATHER_FEATURES.index(f) for f in feats]
            drop = rng.random(len(df)) < rate
            X[np.ix_(drop, cols)] = np.nan
    elif pattern == "block":
        d = daily.copy()
        n_days = d.shape[1]
        run = int(round(rate * n_days))
        starts = rng.integers(0, n_days - run + 1, size=(d.shape[0], d.shape[2]))
        for i in range(d.shape[0]):
            for v in range(d.shape[2]):
                d[i, starts[i, v]: starts[i, v] + run, v] = np.nan
        f = seasonal_features(d, season)
        X[:, :nw] = np.column_stack([f[k] for k in WEATHER_FEATURES])
    else:
        raise ValueError(pattern)
    return X


# ---------------------------------------------------------------- feature selection
def select_features(Xtr, ytr, method, seed):
    """Return column indices chosen on training data only (median-imputed copy)."""
    Xi = SimpleImputer(strategy="median").fit_transform(Xtr)
    if method == "mi":
        mi = mutual_info_regression(Xi, ytr, random_state=seed)
        return np.sort(np.argsort(mi)[::-1][:N_SELECT])
    if method == "rfe":
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.5,
                                   random_state=seed, n_jobs=1)
        sel = RFE(rf, n_features_to_select=N_SELECT, step=2).fit(Xi, ytr)
        return np.flatnonzero(sel.support_)
    raise ValueError(method)


# ---------------------------------------------------------------- one condition
def run_condition(cond, df, X, y):
    """cond: dict with design, model, imputer, seed and optional fs. Returns prediction frame."""
    pred = np.full(len(df), np.nan)
    fold = np.full(len(df), -1)
    for k, (tr, te) in enumerate(splits(cond["design"], df, cond["seed"])):
        Xtr, Xte = X[tr], X[te]
        if cond.get("fs", "full") != "full":
            cols = select_features(Xtr, y[tr], cond["fs"], cond["seed"])
            Xtr, Xte = Xtr[:, cols], Xte[:, cols]
        pred[te] = fit_predict(cond["model"], cond["imputer"], Xtr, y[tr], Xte, cond["seed"])
        fold[te] = k
    keep = fold >= 0
    out = pd.DataFrame({"row_id": df.row_id.to_numpy()[keep], "fold": fold[keep],
                        "y": y[keep], "pred": pred[keep]})
    for k, v in cond.items():
        out[k] = v
    return out


def load_crop(panel, crop):
    df = panel[panel.crop_name == crop].reset_index(drop=True)
    npz = np.load(PROC / f"daily_{crop.lower()}.npz")
    assert (npz["row_id"] == df.row_id.to_numpy()).all()
    return df, npz["daily"], df.season.iloc[0]


def run_grid(tasks, label):
    t0 = time.time()
    print(f"{label}: {len(tasks)} conditions", flush=True)
    parts = Parallel(n_jobs=-1, verbose=0)(delayed(run_condition)(*t) for t in tasks)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RES / f"predictions_{label}.csv.gz", index=False)
    print(f"{label}: done in {time.time() - t0:.0f} s, {len(out):,} predictions", flush=True)


def main(which):
    panel = pd.read_csv(PROC / "panel.csv")
    crops = ["Groundnut", "Bajra", "Wheat"]

    if "E1" in which:
        tasks = []
        for crop in crops:
            df, _, _ = load_crop(panel, crop)
            X, y = df[BASE_FEATURES].to_numpy(float), df.yield_t_ha.to_numpy()
            for design, model, seed in itertools.product(["random", "temporal", "district"],
                                                         ["Mean", "Ridge", "RF", "XGB"], SEEDS):
                tasks.append((dict(crop=crop, design=design, model=model, imputer=None, seed=seed), df, X, y))
        run_grid(tasks, "E1")

    if "E2" in which:
        tasks = []
        combos = [(m, i) for m in ["Ridge", "RF", "XGB"] for i in ["median", "knn", "iterative"]]
        combos.append(("XGB", "native"))
        for crop in crops:
            df, daily, season = load_crop(panel, crop)
            y = df.yield_t_ha.to_numpy()
            for pattern, rate, seed in itertools.product(PATTERNS, RATES, SEEDS):
                X = apply_missingness(df, daily, season, pattern, rate, seed=1000 * seed + int(rate * 100))
                for design, (model, imp) in itertools.product(["random", "temporal", "district"], combos):
                    tasks.append((dict(crop=crop, design=design, model=model, imputer=imp, seed=seed,
                                       pattern=pattern, rate=rate,
                                       missing_frac=float(np.isnan(X[:, :len(WEATHER_FEATURES)]).mean())),
                                  df, X, y))
        run_grid(tasks, "E2")

    if "E3" in which:
        tasks = []
        for crop in crops:
            df, daily, season = load_crop(panel, crop)
            y = df.yield_t_ha.to_numpy()
            for rate, seed in itertools.product([0.0, 0.2, 0.4], SEEDS):
                X = (df[BASE_FEATURES].to_numpy(float) if rate == 0 else
                     apply_missingness(df, daily, season, "mcar", rate, seed=1000 * seed + int(rate * 100)))
                for fs, (model, imp) in itertools.product(["full", "mi", "rfe"],
                                                          [("RF", "median"), ("XGB", "native")]):
                    tasks.append((dict(crop=crop, design="district", model=model, imputer=imp, seed=seed,
                                       pattern="mcar", rate=rate, fs=fs), df, X, y))
        run_grid(tasks, "E3")

    if "E4" in which:
        tasks = []
        for crop in crops:
            df, _, _ = load_crop(panel, crop)
            y = df.yield_t_ha.to_numpy()
            for feats, model, seed in itertools.product(["weather", "weather+location"], ["Ridge", "RF", "XGB"], SEEDS):
                cols = BASE_FEATURES + (["lat", "lon"] if feats == "weather+location" else [])
                tasks.append((dict(crop=crop, design="district", model=model, imputer=None, seed=seed,
                                   features=feats), df, df[cols].to_numpy(float), y))
        run_grid(tasks, "E4")


if __name__ == "__main__":
    main(sys.argv[1:] or ["E1", "E2", "E3", "E4"])
