"""
FireBridge: harmonizing MODIS and VIIRS active-fire hotspots (NASA FIRMS).

Pipeline
  1. Load MODIS (Terra/Aqua, 1 km) and VIIRS (NOAA-21, 375 m) FIRMS CSVs and put them on a common schema.
  2. Match detections in space (MODIS pixel footprint) and time (near-simultaneous overpasses).
  3. Learn the relationship between the sensors:
       a. FRP transfer function  MODIS_FRP ~ f(sum of VIIRS FRP inside the MODIS pixel)
       b. "MODIS detectability" model: P(a VIIRS hotspot is also seen by MODIS)
  4. Build a harmonized 0.25-degree daily grid with MODIS-equivalent counts and FRP from VIIRS,
     and validate it against real MODIS (Aqua) on held-out days.
  5. Export figures, a summary JSON and the data used by the web dashboard (docs/).

Usage:  python harmonize.py  [--modis data/MODIS.csv] [--viirs data/VIIRS.csv]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score

R_EARTH = 6371.0
GRID = 0.25            # degrees
MAX_DT_MIN = 240       # Aqua has drifted ~2-4 h after NOAA-21 (measured), so "same pass" = within 4 h
OUT = "outputs"
DOCS = "docs"


# ---------------------------------------------------------------- loading
def load(path, sensor):
    df = pd.read_csv(path)
    t = df["acq_time"].astype(int).astype(str).str.zfill(4)
    df["time"] = pd.to_datetime(df["acq_date"] + " " + t.str[:2] + ":" + t.str[2:], utc=True)
    if sensor == "MODIS":
        c = df["confidence"].astype(float)
        df["conf"] = np.where(c < 30, "low", np.where(c < 80, "nominal", "high"))
        df["bt_mir"], df["bt_tir"] = df["brightness"], df["bright_t31"]
        df["platform"] = df["satellite"].map({"T": "Terra", "A": "Aqua"})
    else:
        df["conf"] = df["confidence"].map({"l": "low", "n": "nominal", "h": "high"}).fillna(df["confidence"])
        df["bt_mir"], df["bt_tir"] = df["bright_ti4"], df["bright_ti5"]
        df["platform"] = df["satellite"].map({"N": "Suomi-NPP", "N20": "NOAA-20", "1": "NOAA-20",
                                              "N21": "NOAA-21"}).fillna(df["satellite"].astype(str))
    df["sensor"] = sensor
    keep = ["latitude", "longitude", "time", "scan", "track", "frp", "bt_mir", "bt_tir",
            "conf", "daynight", "platform", "sensor"]
    return df[keep].reset_index(drop=True)


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)] * R_EARTH


def region(lat, lon):
    r = np.full(len(lat), "Other", dtype=object)
    r[(lon < -30) & (lat >= 15)] = "North America"
    r[(lon < -30) & (lat < 15)] = "South & Central America"
    r[(lon >= -30) & (lon < 60) & (lat < 37)] = "Africa & Middle East"
    r[(lon >= -30) & (lon < 60) & (lat >= 37)] = "Europe"
    r[(lon >= 60) & (lat >= 37)] = "North Asia"
    r[(lon >= 60) & (lat < 37) & (lat >= -10)] = "South & SE Asia"
    r[(lon >= 100) & (lat < -10)] = "Australia & Oceania"
    return r


# ---------------------------------------------------------------- matching
def match(modis, viirs):
    """For each MODIS pixel, find VIIRS hotspots inside its footprint within MAX_DT_MIN."""
    tree = cKDTree(xyz(viirs.latitude.values, viirs.longitude.values))
    # half-diagonal of the MODIS pixel + half a VIIRS pixel
    radius = 0.5 * np.hypot(modis.scan.values, modis.track.values) + 0.3
    cand = tree.query_ball_point(xyz(modis.latitude.values, modis.longitude.values), r=radius)
    vt = viirs.time.values.astype("datetime64[m]").astype(np.int64)
    mt = modis.time.values.astype("datetime64[m]").astype(np.int64)
    rows = []
    for i, idx in enumerate(cand):
        if not idx:
            continue
        idx = np.asarray(idx)
        dt = np.abs(vt[idx] - mt[i])
        for j, d in zip(idx[dt <= MAX_DT_MIN], dt[dt <= MAX_DT_MIN]):
            rows.append((i, j, d))
    return pd.DataFrame(rows, columns=["m", "v", "dt_min"])


def cotemporal_mask(a, b, max_km=50):
    """True where a detection in `a` had a detection of `b` within max_km and MAX_DT_MIN
    -> both sensors were looking at the same area at about the same time."""
    tree = cKDTree(xyz(b.latitude.values, b.longitude.values))
    cand = tree.query_ball_point(xyz(a.latitude.values, a.longitude.values), r=max_km)
    at = a.time.values.astype("datetime64[m]").astype(np.int64)
    bt = b.time.values.astype("datetime64[m]").astype(np.int64)
    out = np.zeros(len(a), bool)
    for i, idx in enumerate(cand):
        if idx and (np.abs(bt[idx] - at[i]) <= MAX_DT_MIN).any():
            out[i] = True
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modis", default="MODIS_C6_1_Global_7d.csv.gz")
    ap.add_argument("--viirs", default="J2_VIIRS_C2_Global_7d.csv.gz")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DOCS, exist_ok=True)

    modis, viirs = load(args.modis, "MODIS"), load(args.viirs, "VIIRS")
    print(f"MODIS {len(modis):,} | VIIRS {len(viirs):,}")
    S = {"n_modis": len(modis), "n_viirs": len(viirs),
         "period": [str(min(modis.time.min(), viirs.time.min()).date()),
                    str(max(modis.time.max(), viirs.time.max()).date())],
         "viirs_platform": viirs.platform.mode()[0]}

    # ---- 1. matching
    pairs = match(modis, viirs)
    m_co = cotemporal_mask(modis, viirs)              # MODIS pixels in overlapping passes
    v_co = cotemporal_mask(viirs, modis)              # VIIRS hotspots in overlapping passes
    modis["matched"] = False
    modis.loc[pairs.m.unique(), "matched"] = True
    viirs["matched"] = False
    viirs.loc[pairs.v.unique(), "matched"] = True
    S["pairs"] = len(pairs)
    S["median_dt_min"] = float(pairs.dt_min.median())
    S["modis_cotemporal"] = int(m_co.sum())
    S["viirs_cotemporal"] = int(v_co.sum())
    S["pct_modis_confirmed_by_viirs"] = float(modis.matched[m_co].mean() * 100)
    S["pct_viirs_seen_by_modis"] = float(viirs.matched[v_co].mean() * 100)
    S["viirs_per_modis_pixel"] = float(pairs.groupby("m").size().mean())
    print(f"pairs {len(pairs):,}; MODIS confirmed by VIIRS {S['pct_modis_confirmed_by_viirs']:.1f}%, "
          f"VIIRS seen by MODIS {S['pct_viirs_seen_by_modis']:.1f}%")

    # ---- 2a. FRP transfer function (MODIS pixel vs VIIRS hotspots inside it)
    agg = pairs.assign(vfrp=viirs.frp.values[pairs.v]).groupby("m").agg(vfrp=("vfrp", "sum"),
                                                                         nv=("v", "size"))
    agg["mfrp"] = modis.frp.values[agg.index]
    agg = agg[(agg.vfrp > 0) & (agg.mfrp > 0)]
    X, y = np.log10(agg[["vfrp"]].values), np.log10(agg.mfrp.values)
    hub = HuberRegressor().fit(X, y)
    pred = hub.predict(X)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    S["frp_fit"] = {"slope": float(hub.coef_[0]), "intercept": float(hub.intercept_), "r2": float(r2),
                    "n": int(len(agg)), "ratio_sum_modis_over_viirs": float(agg.mfrp.sum() / agg.vfrp.sum())}
    print("FRP fit", S["frp_fit"])

    # ---- 2b. MODIS detectability model (trained on early days, tested on the last days)
    feats = ["frp", "bt_mir", "bt_tir", "scan", "track", "night", "conf_n", "absl"]
    viirs["night"] = (viirs.daynight == "N").astype(int)
    viirs["conf_n"] = viirs.conf.map({"low": 0, "nominal": 1, "high": 2}).fillna(1)
    viirs["absl"] = viirs.latitude.abs()
    co = viirs[v_co]
    days = np.sort(co.time.dt.date.unique())
    test_days = set(days[-2:])
    is_test = co.time.dt.date.isin(test_days)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
    clf.fit(co.loc[~is_test, feats], co.loc[~is_test, "matched"])
    p_test = clf.predict_proba(co.loc[is_test, feats])[:, 1]
    S["detect_model"] = {"auc_test": float(roc_auc_score(co.loc[is_test, "matched"], p_test)),
                         "test_days": [str(d) for d in sorted(test_days)],
                         "pred_vs_obs_matched": [float(p_test.sum()), int(co.loc[is_test, "matched"].sum())]}
    viirs["p_modis"] = clf.predict_proba(viirs[feats])[:, 1]
    print("detect model", S["detect_model"])

    # ---- 3. harmonized grid
    k = S["viirs_per_modis_pixel"]
    viirs["modis_eq"] = viirs.p_modis / k                       # MODIS-equivalent pixel count
    fs = S["frp_fit"]["ratio_sum_modis_over_viirs"]
    viirs["frp_eq"] = viirs.frp * fs * viirs.p_modis              # expected FRP MODIS would report
    for d in (modis, viirs):
        d["gy"] = np.floor(d.latitude / GRID).astype(int)
        d["gx"] = np.floor(d.longitude / GRID).astype(int)
        d["day"] = d.time.dt.strftime("%Y-%m-%d")
    aqua = modis[modis.platform == "Aqua"]
    gm = aqua.groupby(["day", "gy", "gx"]).agg(modis_n=("frp", "size"), modis_frp=("frp", "sum"))
    gv = viirs.groupby(["day", "gy", "gx"]).agg(
        viirs_n=("frp", "size"), viirs_frp=("frp", "sum"), frp_max=("frp", "max"), p_sum=("p_modis", "sum"),
        bt_max=("bt_mir", "max"), night=("night", "mean"), high=("conf_n", lambda c: (c == 2).mean()),
        absl=("absl", "mean"), det_eq=("modis_eq", "sum"))
    g = gm.join(gv, how="outer").fillna(0).reset_index()
    # Cell-level transfer model: VIIRS cell statistics -> MODIS (Aqua) count & FRP.
    # Trained on the first days, validated on the last 2 days (never seen in training).
    cf = ["viirs_n", "viirs_frp", "frp_max", "p_sum", "bt_max", "night", "high", "absl"]
    test_mask = g.day.isin([str(d) for d in test_days]).values
    has_v = (g.viirs_n > 0).values
    reg_n = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.06, random_state=42)
    reg_n.fit(g.loc[has_v & ~test_mask, cf], g.loc[has_v & ~test_mask, "modis_n"])
    reg_f = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.06, random_state=42)
    reg_f.fit(np.log1p(g.loc[has_v & ~test_mask, cf]), g.loc[has_v & ~test_mask, "modis_frp"])
    g["eq_n"], g["eq_frp"] = 0.0, 0.0
    g.loc[has_v, "eq_n"] = reg_n.predict(g.loc[has_v, cf])
    g.loc[has_v, "eq_frp"] = reg_f.predict(np.log1p(g.loc[has_v, cf]))
    val = g[test_mask]

    def stats(a, b):
        return {"r": float(np.corrcoef(a, b)[0, 1]), "bias_pct": float((a.sum() - b.sum()) / b.sum() * 100),
                "mae": float(np.abs(a - b).mean())}
    S["validation"] = {"cells": int(len(val)), "days": [str(d) for d in sorted(test_days)],
                       "note": "all 0.25deg cell-days with any MODIS-Aqua or VIIRS fire on held-out days",
                       "raw_viirs_count": stats(val.viirs_n, val.modis_n),
                       "harmonized_count": stats(val.eq_n, val.modis_n),
                       "raw_viirs_frp": stats(val.viirs_frp, val.modis_frp),
                       "harmonized_frp": stats(val.eq_frp, val.modis_frp)}
    v1 = val.assign(y1=val.gy // 4, x1=val.gx // 4).groupby(["y1", "x1"])[
        ["modis_n", "viirs_n", "eq_n"]].sum()                      # 1-degree, 2-day totals
    S["validation"]["1deg_raw_viirs_count"] = stats(v1.viirs_n, v1.modis_n)
    S["validation"]["1deg_harmonized_count"] = stats(v1.eq_n, v1.modis_n)
    print("validation", json.dumps(S["validation"], indent=1))

    # ---- regional table
    modis["region"] = region(modis.latitude.values, modis.longitude.values)
    viirs["region"] = region(viirs.latitude.values, viirs.longitude.values)
    g["region"] = region((g.gy.values + 0.5) * GRID, (g.gx.values + 0.5) * GRID)
    reg = []
    for r in sorted(set(modis.region) | set(viirs.region)):
        mm, vv = modis[(modis.region == r) & m_co], viirs[(viirs.region == r) & v_co]
        reg.append({"region": r, "modis": int((modis.region == r).sum()), "viirs": int((viirs.region == r).sum()),
                    "viirs_modis_eq": float(g.loc[g.region == r, "eq_n"].sum()),
                    "modis_aqua": int(((modis.region == r) & (modis.platform == "Aqua")).sum()),
                    "pct_modis_confirmed": float(mm.matched.mean() * 100) if len(mm) else None,
                    "pct_viirs_seen": float(vv.matched.mean() * 100) if len(vv) else None})
    S["regions"] = reg

    # ---- daily series (global)
    daily = g.groupby("day").agg(modis_aqua=("modis_n", "sum"), viirs=("viirs_n", "sum"),
                                 viirs_eq=("eq_n", "sum")).reset_index()
    S["daily"] = daily.to_dict("records")

    # ---- figures
    figures(agg, hub, viirs, co, is_test, p_test, val, daily, S)

    # ---- dashboard data: compact grid (all days combined) + sample of points
    with open(f"{DOCS}/summary.json", "w") as f:
        json.dump(S, f, indent=1, default=str)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(S, f, indent=1, default=str)
    # per-day cells for the time slider
    day_list = sorted(g.day.unique())
    di = {d: i for i, d in enumerate(day_list)}
    rows = [[di[d], int(y), int(x), int(m), int(v), round(float(e), 2)]
            for d, y, x, m, v, e in zip(g.day, g.gy, g.gx, g.modis_n, g.viirs_n, g.eq_n)]
    with open(f"{DOCS}/days.json", "w") as f:
        json.dump({"grid": GRID, "days": day_list, "cols": ["day", "gy", "gx", "modis_n", "viirs_n", "eq_n"],
                   "rows": rows}, f, separators=(",", ":"))
    # curves for the interactive charts
    bins = np.logspace(np.log10(0.3), np.log10(300), 22)
    b = pd.cut(co.frp, bins)
    cur = co.assign(p=viirs.loc[co.index, "p_modis"]).groupby(b, observed=True).agg(
        obs=("matched", "mean"), mod=("p", "mean"), n=("frp", "size"))
    # conditional curves for the simulator: FRP x day/night x scan position (nadir vs swath edge)
    bins2 = np.logspace(np.log10(0.3), np.log10(300), 12)
    cc = co.assign(p=viirs.loc[co.index, "p_modis"], b=pd.cut(co.frp, bins2), edge=(co.scan > 0.5).astype(int))
    cond = {}
    for (nt, ed), sub in cc.groupby(["night", "edge"]):
        t = sub.groupby("b", observed=True).agg(obs=("matched", "mean"), mod=("p", "mean"), n=("frp", "size"))
        t = t[t.n >= 25]
        cond[f"{'night' if nt else 'day'}_{'edge' if ed else 'nadir'}"] = [
            {"frp": float(iv.mid), "obs": float(r.obs), "mod": float(r["mod"]), "n": int(r.n)} for iv, r in t.iterrows()]
    dth = np.histogram(pairs.dt_min, bins=np.arange(0, MAX_DT_MIN + 1, 15))
    with open(f"{DOCS}/curves.json", "w") as f:
        json.dump({"detect": [{"frp": float(iv.mid), "obs": float(r.obs), "mod": float(r["mod"]), "n": int(r.n)}
                              for iv, r in cur.iterrows()],
                   "dt_hist": {"edges": dth[1].tolist(), "counts": dth[0].tolist()},
                   "detect_cond": cond}, f)
    g.to_csv(f"{OUT}/harmonized_grid_daily.csv.gz", index=False, compression="gzip")
    g[["day", "gy", "gx", "modis_n", "modis_frp", "viirs_n", "viirs_frp", "eq_n", "eq_frp"]].round(3).to_csv(
        f"{DOCS}/firebridge_harmonized_grid.csv.gz", index=False, compression="gzip")
    print("done ->", OUT, DOCS)


def figures(agg, hub, viirs, co, is_test, p_test, val, daily, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, MUTE = "#0b0920", "#ece9ff", "#9b95c9"
    plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": INK,
                         "axes.edgecolor": "#3a3466", "xtick.color": MUTE, "ytick.color": MUTE,
                         "axes.titlecolor": INK, "legend.labelcolor": INK})
    C_M, C_V, C_H = "#ff7a45", "#22d3ee", "#c084fc"

    # FRP transfer
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.hexbin(np.log10(agg.vfrp), np.log10(agg.mfrp), gridsize=50, mincnt=1, cmap="plasma", bins="log")
    xs = np.linspace(np.log10(agg.vfrp).min(), np.log10(agg.vfrp).max(), 50)
    ax.plot(xs, hub.predict(xs[:, None]), color=C_V, lw=2, label=f"fit (slope {hub.coef_[0]:.2f}, R² {S['frp_fit']['r2']:.2f})")
    ax.plot(xs, xs, "--", color=MUTE, lw=1, label="1:1")
    ax.set_xlabel("log10 Σ VIIRS FRP inside MODIS pixel (MW)")
    ax.set_ylabel("log10 MODIS FRP (MW)")
    ax.set_title("FRP transfer function MODIS ↔ VIIRS")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_frp_transfer.png"); plt.close(fig)

    # detection probability vs FRP
    fig, ax = plt.subplots(figsize=(5, 4.2))
    bins = np.logspace(np.log10(max(co.frp.min(), 0.1)), np.log10(co.frp.quantile(0.995)), 25)
    b = pd.cut(co.frp, bins)
    obs = co.groupby(b, observed=True).matched.mean()
    mod = co.assign(p=viirs.loc[co.index, "p_modis"]).groupby(b, observed=True).p.mean()
    mids = [iv.mid for iv in obs.index]
    ax.semilogx(mids, obs.values * 100, "o", color=C_M, label="observed")
    ax.semilogx(mids, mod.values * 100, "-", color=C_V, lw=2, label="model")
    ax.set_xlabel("VIIRS hotspot FRP (MW)"); ax.set_ylabel("% also detected by MODIS")
    ax.set_title(f"MODIS detectability (test AUC {S['detect_model']['auc_test']:.2f})")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_detectability.png"); plt.close(fig)

    # validation scatter
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    lim = max(val.modis_n.max(), 1)
    for ax, col, ttl, c in [(axs[0], "viirs_n", "Raw VIIRS count", C_V), (axs[1], "eq_n", "Harmonized (MODIS-equivalent)", C_H)]:
        st = S["validation"]["raw_viirs_count" if col == "viirs_n" else "harmonized_count"]
        ax.scatter(val.modis_n, val[col], s=6, alpha=0.4, color=c)
        ax.plot([0, lim], [0, lim], "--", color=MUTE, lw=1)
        ax.set_xlabel("Real MODIS Aqua fire pixels per cell-day")
        ax.set_title(f"{ttl}\nr={st['r']:.2f}, bias={st['bias_pct']:+.0f}%")
    axs[0].set_ylabel("VIIRS-derived count")
    axs[1].set_ylim(0, lim * 1.1); axs[0].set_ylim(0, None)
    fig.suptitle("Validation on held-out days (0.25° cells)", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_validation.png", bbox_inches="tight"); plt.close(fig)

    # daily series
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = pd.to_datetime(daily.day)
    ax.plot(x, daily.modis_aqua, "o-", color=C_M, label="MODIS Aqua (real)")
    ax.plot(x, daily.viirs, "s-", color=C_V, label="VIIRS raw")
    ax.plot(x, daily.viirs_eq, "^-", color=C_H, label="VIIRS harmonized → MODIS-equivalent")
    ax.set_yscale("log"); ax.set_ylabel("Global fire detections / day")
    ax.set_title("Continuity: VIIRS translated to the MODIS record")
    ax.legend(frameon=False, fontsize=8); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_daily.png"); plt.close(fig)


if __name__ == "__main__":
    main()
