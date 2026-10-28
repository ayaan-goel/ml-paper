"""Compute metrics, statistical tests, tables and figures from saved predictions.

Usage: python src/analysis.py
Reads   results/predictions_E*.csv.gz, data/processed/panel.csv
Writes  results/tables/*.csv, paper/tables/*.tex, results/figures/*.pdf
"""
import pathlib, sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from experiments import BASE_FEATURES, N_FOLDS, load_crop, select_features, splits

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, TAB, FIG = ROOT / "results", ROOT / "results/tables", ROOT / "results/figures"
PTAB = ROOT / "paper/tables"
for d in (TAB, FIG, PTAB):
    d.mkdir(parents=True, exist_ok=True)

CROPS = ["Groundnut", "Bajra", "Wheat"]
DESIGN_NAMES = {"random": "Random split", "temporal": "Temporal", "district": "District holdout"}
PATTERN_NAMES = {"mcar": "Random (MCAR)", "variable": "Variable-wise", "block": "Block outage"}
N_BOOT = 2000
rng = np.random.default_rng(2026)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150})


# ---------------------------------------------------------------- helpers
def metrics(y, p):
    e = p - y
    sse = (e ** 2).sum()
    return dict(RMSE=np.sqrt((e ** 2).mean()), MAE=np.abs(e).mean(),
                R2=1 - sse / ((y - y.mean()) ** 2).sum(), nRMSE=100 * np.sqrt((e ** 2).mean()) / y.mean())


def summarise(pred, keys):
    """Metrics per condition and seed, then mean and sd over seeds."""
    rows = []
    for k, g in pred.groupby(keys + ["seed"], dropna=False):
        rows.append(dict(zip(keys + ["seed"], k), **metrics(g.y.to_numpy(), g.pred.to_numpy())))
    m = pd.DataFrame(rows)
    agg = m.groupby(keys, dropna=False)[["RMSE", "MAE", "R2", "nRMSE"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def crop_mean(df, keys):
    """Average a per-crop summary over crops (nRMSE and R2 are comparable across crops)."""
    return df.groupby(keys, dropna=False)[["nRMSE_mean", "R2_mean"]].mean().reset_index()


def tex_table(df, path, colfmt, header, caption=None):
    lines = [r"\begin{tabular}{" + colfmt + "}", r"\toprule", header + r" \\", r"\midrule"]
    for row in df:
        lines.append(row if row.startswith("\\") else row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (PTAB / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def districts_of(panel):
    return panel.set_index(["crop_name", "row_id"]).district_name


def boot_rmse_diff(a, b, groups, n=N_BOOT):
    """Bootstrap over districts: CI for RMSE(a) - RMSE(b); a, b are squared-error arrays aligned by row."""
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    d = np.empty(n)
    for i in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        d[i] = np.sqrt(a[s].mean()) - np.sqrt(b[s].mean())
    return np.sqrt(a.mean()) - np.sqrt(b.mean()), np.percentile(d, [2.5, 97.5])


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "$-$") if np.isfinite(x) else "--"


# ---------------------------------------------------------------- dataset table
def table_dataset(panel):
    rows, out = [], []
    windows = {"Kharif": "1 Jun--31 Oct", "Rabi": "1 Nov--31 Mar"}
    for c in CROPS:
        s = panel[panel.crop_name == c]
        r = dict(crop=c, season=s.season.iloc[0], obs=len(s), districts=s.district_name.nunique(),
                 years=f"{s.year_start.min()}--{s.year_start.max() + 1}", mean=s.yield_t_ha.mean(),
                 sd=s.yield_t_ha.std())
        out.append(r)
        rows.append(f"{c} & {r['season']} ({windows[r['season']]}) & {r['obs']} & {r['districts']} & "
                    f"{r['years']} & {r['mean']:.2f} $\\pm$ {r['sd']:.2f}")
    pd.DataFrame(out).to_csv(TAB / "dataset_summary.csv", index=False)
    tex_table(rows, "dataset.tex", "llrrcc",
              r"Crop & Season (window) & Obs. & Distr. & Years & Yield (t/ha)")


# ---------------------------------------------------------------- E1
def analyse_e1(panel):
    p = pd.read_csv(RES / "predictions_E1.csv.gz")
    s = summarise(p, ["crop", "design", "model"])
    s.to_csv(TAB / "E1_per_crop.csv", index=False)

    # per-crop table: R2 and RMSE for each design
    models = ["Mean", "Ridge", "RF", "XGB"]
    rows = []
    for c in CROPS:
        rows.append(r"\multicolumn{7}{l}{\textit{" + c + r"}} \\")
        for m in models:
            cells = []
            for d in ["random", "temporal", "district"]:
                r = s[(s.crop == c) & (s.design == d) & (s.model == m)].iloc[0]
                cells += [fmt(r.RMSE_mean, 3), fmt(r.R2_mean)]
            rows.append(f"{m} & " + " & ".join(cells))
    tex_table(rows, "e1.tex", "lrrrrrr",
              r" & \multicolumn{2}{c}{Random split} & \multicolumn{2}{c}{Temporal} & "
              r"\multicolumn{2}{c}{District holdout} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
              r"Model & RMSE & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$")

    # optimism of random split vs district holdout: bootstrap over districts per crop, seed 0
    dist = districts_of(panel)
    out = []
    for c in CROPS:
        for m in ["Ridge", "RF", "XGB"]:
            a = p[(p.crop == c) & (p.model == m) & (p.design == "district") & (p.seed == 0)].sort_values("row_id")
            b = p[(p.crop == c) & (p.model == m) & (p.design == "random") & (p.seed == 0)].sort_values("row_id")
            assert (a.row_id.values == b.row_id.values).all()
            g = dist.loc[c].loc[a.row_id].to_numpy()
            d, ci = boot_rmse_diff(((a.pred - a.y) ** 2).to_numpy(), ((b.pred - b.y) ** 2).to_numpy(), g)
            out.append(dict(crop=c, model=m, rmse_gap=d, ci_lo=ci[0], ci_hi=ci[1],
                            rel_gap_pct=100 * d / np.sqrt(((b.pred - b.y) ** 2).mean())))
    pd.DataFrame(out).to_csv(TAB / "E1_random_vs_district_bootstrap.csv", index=False)

    # figure: R2 by design per crop
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    colors = {"random": "#4C72B0", "temporal": "#DD8452", "district": "#55A868"}
    models = ["Ridge", "RF", "XGB"]  # the mean predictor is reported in the table only
    for ax, c in zip(axes, CROPS):
        x = np.arange(len(models))
        for j, d in enumerate(["random", "temporal", "district"]):
            v = [s[(s.crop == c) & (s.design == d) & (s.model == m)].R2_mean.iloc[0] for m in models]
            e = [s[(s.crop == c) & (s.design == d) & (s.model == m)].R2_std.iloc[0] for m in models]
            ax.bar(x + (j - 1) * 0.27, v, 0.27, yerr=e, color=colors[d], label=DESIGN_NAMES[d],
                   error_kw=dict(lw=0.6, capsize=1.5))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, models)
        ax.set_title(c)
    axes[0].set_ylabel("$R^2$")
    axes[0].set_ylim(-0.25, 0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(FIG / "fig_e1_designs.pdf", bbox_inches="tight")
    plt.close(fig)
    return s


# ---------------------------------------------------------------- E2
STRATS = [("Ridge", "median"), ("Ridge", "knn"), ("Ridge", "iterative"),
          ("RF", "median"), ("RF", "knn"), ("RF", "iterative"),
          ("XGB", "median"), ("XGB", "knn"), ("XGB", "iterative"), ("XGB", "native")]


def strat_label(m, i):
    return f"{m} + {'native' if i == 'native' else {'median': 'median', 'knn': 'KNN', 'iterative': 'iterative'}[i]}"


def analyse_e2(panel, e1):
    p = pd.read_csv(RES / "predictions_E2.csv.gz")
    s = summarise(p, ["crop", "design", "pattern", "rate", "model", "imputer"])
    s.to_csv(TAB / "E2_per_crop.csv", index=False)
    mf = p.groupby(["pattern", "rate"]).missing_frac.mean().reset_index()
    mf.to_csv(TAB / "E2_missing_fraction.csv", index=False)

    # complete-data reference per crop/design/model (E1)
    ref = e1.set_index(["crop", "design", "model"])
    s["ref_nRMSE"] = [ref.loc[(r.crop, r.design, r.model), "nRMSE_mean"] for r in s.itertuples()]
    s["ref_R2"] = [ref.loc[(r.crop, r.design, r.model), "R2_mean"] for r in s.itertuples()]
    s["d_nRMSE"] = s.nRMSE_mean - s.ref_nRMSE
    cm = s.groupby(["design", "pattern", "rate", "model", "imputer"])[
        ["nRMSE_mean", "R2_mean", "d_nRMSE"]].mean().reset_index()
    cm.to_csv(TAB / "E2_crop_mean.csv", index=False)

    # table: district holdout, rate 0.3, all strategies x patterns
    rows = []
    for m, i in STRATS:
        cells = []
        for pat in ["mcar", "variable", "block"]:
            r = cm[(cm.design == "district") & (cm.pattern == pat) & (cm.rate == 0.3) &
                   (cm.model == m) & (cm.imputer == i)].iloc[0]
            cells += [fmt(r.d_nRMSE, 1), fmt(r.R2_mean)]
        rows.append(strat_label(m, i) + " & " + " & ".join(cells))
    tex_table(rows, "e2_strategies.tex", "lrrrrrr",
              r" & \multicolumn{2}{c}{Random (MCAR)} & \multicolumn{2}{c}{Variable-wise} & "
              r"\multicolumn{2}{c}{Block outage} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
              r"Strategy & $\Delta$nRMSE & $R^2$ & $\Delta$nRMSE & $R^2$ & $\Delta$nRMSE & $R^2$")

    # average rank of strategies over (crop, pattern, rate) blocks, district holdout + Friedman
    blocks = s[s.design == "district"].copy()
    blocks["strategy"] = [strat_label(m, i) for m, i in zip(blocks.model, blocks.imputer)]
    piv = blocks.pivot_table(index=["crop", "pattern", "rate"], columns="strategy", values="RMSE_mean")
    ranks = piv.rank(axis=1)
    fr = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    rk = ranks.mean().sort_values()
    rk.to_csv(TAB / "E2_strategy_ranks.csv")
    by_pat = {pat: ranks.xs(pat, level="pattern").mean() for pat in ["mcar", "variable", "block"]}
    pd.DataFrame(by_pat).to_csv(TAB / "E2_strategy_ranks_by_pattern.csv")
    (TAB / "E2_friedman.txt").write_text(f"Friedman chi2={fr.statistic:.2f}, p={fr.pvalue:.3g}, "
                                         f"blocks={len(piv)}, strategies={piv.shape[1]}\n")

    # interaction of missingness and geographic shift (H3), bootstrap over districts, seed 0, rate 0.3
    dist = districts_of(panel)
    e1p = pd.read_csv(RES / "predictions_E1.csv.gz")
    out = []
    for c in CROPS:
        for m, i in [("Ridge", "median"), ("RF", "median"), ("RF", "iterative"), ("XGB", "native"), ("XGB", "median")]:
            for pat in ["mcar", "variable", "block"]:
                for sd in [0, 1, 2]:
                    def se(df):
                        df = df.sort_values("row_id")
                        return df.row_id.to_numpy(), ((df.pred - df.y) ** 2).to_numpy()
                    ids, rc = se(e1p[(e1p.crop == c) & (e1p.model == m) & (e1p.design == "random") & (e1p.seed == sd)])
                    _, dc = se(e1p[(e1p.crop == c) & (e1p.model == m) & (e1p.design == "district") & (e1p.seed == sd)])
                    q = p[(p.crop == c) & (p.model == m) & (p.imputer == i) & (p.pattern == pat) &
                          (p.rate == 0.3) & (p.seed == sd)]
                    _, rm = se(q[q.design == "random"])
                    _, dm = se(q[q.design == "district"])
                    ymean = panel[panel.crop_name == c].yield_t_ha.mean()
                    r = lambda a: 100 * np.sqrt(a.mean()) / ymean
                    geo, miss, comb = r(dc) - r(rc), r(rm) - r(rc), r(dm) - r(rc)
                    out.append(dict(crop=c, model=m, imputer=i, pattern=pat, seed=sd, geo=geo, miss=miss,
                                    combined=comb, interaction=comb - geo - miss))
    inter = pd.DataFrame(out)
    inter.to_csv(TAB / "E2_interaction_per_seed.csv", index=False)
    ig = inter.groupby(["model", "imputer", "pattern"])[["geo", "miss", "combined", "interaction"]].agg(["mean", "std"])
    ig.to_csv(TAB / "E2_interaction.csv")
    t = stats.ttest_1samp(inter.groupby(["crop", "model", "imputer", "pattern"]).interaction.mean(), 0)
    (TAB / "E2_interaction_test.txt").write_text(
        f"one-sample t-test of interaction (crop x strategy x pattern means) vs 0: "
        f"mean={inter.interaction.mean():.3f}, t={t.statistic:.2f}, p={t.pvalue:.3g}, n={t.df + 1}\n")

    rows = []
    for (m, i) in [("Ridge", "median"), ("RF", "median"), ("RF", "iterative"), ("XGB", "median"), ("XGB", "native")]:
        for pat in ["mcar", "variable", "block"]:
            g = inter[(inter.model == m) & (inter.imputer == i) & (inter.pattern == pat)]
            gm = g.groupby("crop")[["geo", "miss", "combined", "interaction"]].mean().mean()
            rows.append(f"{strat_label(m, i)} & {PATTERN_NAMES[pat]} & {fmt(gm.geo, 1)} & {fmt(gm.miss, 1)} & "
                        f"{fmt(gm.combined, 1)} & {fmt(gm.interaction, 1)}")
    tex_table(rows, "e2_interaction.tex", "llrrrr",
              r"Strategy & Pattern & Geo. & Miss. & Comb. & Inter.")

    # figure: degradation curves, district holdout, crop mean
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    styles = {("Ridge", "median"): ("#8172B3", "-", "o"), ("RF", "median"): ("#55A868", "-", "s"),
              ("RF", "iterative"): ("#55A868", "--", "s"), ("XGB", "median"): ("#C44E52", "-", "^"),
              ("XGB", "knn"): ("#C44E52", ":", "^"), ("XGB", "native"): ("#C44E52", "--", "v")}
    for ax, pat in zip(axes, ["mcar", "variable", "block"]):
        for (m, i), (col, ls, mk) in styles.items():
            q = cm[(cm.design == "district") & (cm.pattern == pat) & (cm.model == m) & (cm.imputer == i)].sort_values("rate")
            ax.plot([0] + list(100 * q.rate), [0] + list(q.d_nRMSE), color=col, ls=ls, marker=mk, ms=3,
                    lw=1, label=strat_label(m, i))
        ax.set_title(PATTERN_NAMES[pat])
        ax.set_xlabel("Missingness rate (%)")
        ax.set_xticks([0, 10, 20, 30, 40])
    axes[0].set_ylabel(r"$\Delta$nRMSE (pp)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.savefig(FIG / "fig_e2_degradation.pdf", bbox_inches="tight")
    plt.close(fig)
    return s, cm, rk, fr


# ---------------------------------------------------------------- E3, E4
def analyse_e3(panel):
    p = pd.read_csv(RES / "predictions_E3.csv.gz")
    s = summarise(p, ["crop", "model", "imputer", "rate", "fs"])
    s.to_csv(TAB / "E3_per_crop.csv", index=False)
    cm = crop_mean(s, ["model", "imputer", "rate", "fs"])
    cm.to_csv(TAB / "E3_crop_mean.csv", index=False)
    rows = []
    names = {"full": f"Full ({len(BASE_FEATURES)})", "mi": "Mutual info. (8)", "rfe": "RFE (8)"}
    for m, i in [("RF", "median"), ("XGB", "native")]:
        for fs in ["full", "mi", "rfe"]:
            cells = []
            for rate in [0.0, 0.2, 0.4]:
                r = cm[(cm.model == m) & (cm.fs == fs) & (cm.rate == rate)].iloc[0]
                cells += [fmt(r.nRMSE_mean, 1), fmt(r.R2_mean)]
            rows.append(f"{strat_label(m, i)} & {names[fs]} & " + " & ".join(cells))
    tex_table(rows, "e3.tex", "llrrrrrr",
              r" & & \multicolumn{2}{c}{0\%} & \multicolumn{2}{c}{20\%} & \multicolumn{2}{c}{40\%} \\ "
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
              r"Strategy & Features & nRMSE & $R^2$ & nRMSE & $R^2$ & nRMSE & $R^2$")

    # which features are selected (complete data, seed 0, district folds)
    counts = {}
    for c in CROPS:
        df, _, _ = load_crop(panel, c)
        X, y = df[BASE_FEATURES].to_numpy(float), df.yield_t_ha.to_numpy()
        for tr, _ in splits("district", df, 0):
            for meth in ["mi", "rfe"]:
                for j in select_features(X[tr], y[tr], meth, 0):
                    counts[(meth, BASE_FEATURES[j])] = counts.get((meth, BASE_FEATURES[j]), 0) + 1
    sel = pd.Series(counts).unstack(0).fillna(0).astype(int)
    sel.sort_values("rfe", ascending=False).to_csv(TAB / "E3_selected_features.csv")
    return cm


def analyse_e4():
    p = pd.read_csv(RES / "predictions_E4.csv.gz")
    s = summarise(p, ["crop", "model", "features"])
    s.to_csv(TAB / "E4_per_crop.csv", index=False)
    cm = crop_mean(s, ["model", "features"])
    cm.to_csv(TAB / "E4_crop_mean.csv", index=False)
    rows = []
    for m in ["Ridge", "RF", "XGB"]:
        w = s[(s.model == m) & (s.features == "weather")].set_index("crop")
        l = s[(s.model == m) & (s.features == "weather+location")].set_index("crop")
        cells = [f"{fmt(w.loc[c, 'R2_mean'])} " + r"$\rightarrow$" + f" {fmt(l.loc[c, 'R2_mean'])}" for c in CROPS]
        rows.append(f"{m} & " + " & ".join(cells))
    tex_table(rows, "e4.tex", "lccc", "Model & " + " & ".join(CROPS))
    return cm


# ---------------------------------------------------------------- comparison with prior-style evaluation
def table_prior_vs_ours(e1, cm2, rk):
    """R2 under the usual random-split protocol vs realistic conditions (crop means)."""
    e1c = crop_mean(e1, ["design", "model"]).set_index(["design", "model"])
    strategies = [("Ridge", "median"), ("XGB", "median"), ("XGB", "native"), ("RF", "median"), ("RF", "iterative")]
    cols = ["random", "temporal", "district", "mcar", "variable", "block"]
    out = []
    for m, i in strategies:
        r = dict(strategy=strat_label(m, i), random=e1c.loc[("random", m), "R2_mean"],
                 temporal=e1c.loc[("temporal", m), "R2_mean"], district=e1c.loc[("district", m), "R2_mean"])
        for pat in ["mcar", "variable", "block"]:
            r[pat] = cm2[(cm2.design == "district") & (cm2.pattern == pat) & (cm2.rate == 0.3) &
                         (cm2.model == m) & (cm2.imputer == i)].R2_mean.iloc[0]
        r["optimism"] = r["random"] - np.mean([r["mcar"], r["variable"], r["block"]])
        out.append(r)
    df = pd.DataFrame(out)
    df.to_csv(TAB / "prior_vs_realistic.csv", index=False)
    best = {c: df[c].max() for c in cols}
    rows = []
    for r in df.itertuples():
        cells = [(r"\textbf{" + fmt(getattr(r, c)) + "}") if np.isclose(getattr(r, c), best[c]) else fmt(getattr(r, c))
                 for c in cols]
        rows.append(f"{r.strategy} & " + " & ".join(cells) + f" & {fmt(r.optimism)}")
    tex_table(rows, "prior_vs_ours.tex", "lrrrrrrr",
              r" & Usual & \multicolumn{2}{c}{Complete data} & \multicolumn{3}{c}{Distr.\ + 30\% missing} & \\ "
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}"
              r"Strategy & random & Temp. & Distr. & MCAR & Var. & Block & Gap")
    return df


def main():
    panel = pd.read_csv(ROOT / "data/processed/panel.csv")
    table_dataset(panel)
    e1 = analyse_e1(panel)
    print("E1 done")
    if (RES / "predictions_E2.csv.gz").exists():
        s2, cm2, rk, fr = analyse_e2(panel, e1)
        print("E2 done; strategy ranks:\n", rk.round(2), "\n", fr)
        table_prior_vs_ours(e1, cm2, rk)
        compare_recommended(panel)
    if (RES / "predictions_E3.csv.gz").exists():
        analyse_e3(panel)
        print("E3 done")
    if (RES / "predictions_E4.csv.gz").exists():
        analyse_e4()
        print("E4 done")




# ---------------------------------------------------------------- recommended vs alternatives
def compare_recommended(panel, rate=0.3):
    """Bootstrap over districts: RMSE(alternative) - RMSE(RF + iterative), district holdout."""
    p = pd.read_csv(RES / "predictions_E2.csv.gz")
    p = p[(p.design == "district") & (p.rate == rate)]
    dist = districts_of(panel)
    out = []
    for c in CROPS:
        for pat in ["mcar", "variable", "block"]:
            base = p[(p.crop == c) & (p.pattern == pat) & (p.model == "RF") & (p.imputer == "iterative")
                     & (p.seed == 0)].sort_values("row_id")
            g = dist.loc[c].loc[base.row_id].to_numpy()
            eb = ((base.pred - base.y) ** 2).to_numpy()
            for m, i in [("XGB", "native"), ("XGB", "median"), ("Ridge", "median"), ("RF", "median")]:
                alt = p[(p.crop == c) & (p.pattern == pat) & (p.model == m) & (p.imputer == i)
                        & (p.seed == 0)].sort_values("row_id")
                d, ci = boot_rmse_diff(((alt.pred - alt.y) ** 2).to_numpy(), eb, g)
                out.append(dict(crop=c, pattern=pat, alternative=strat_label(m, i), rmse_diff=d,
                                ci_lo=ci[0], ci_hi=ci[1], significant=bool(ci[0] > 0 or ci[1] < 0)))
    df = pd.DataFrame(out)
    df.to_csv(TAB / "recommended_vs_alternatives.csv", index=False)
    return df


if __name__ == "__main__":
    main()
