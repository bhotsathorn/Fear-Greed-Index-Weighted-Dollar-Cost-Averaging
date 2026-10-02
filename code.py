import os
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import ssl
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BASELINE = 1000.0
SLIPPAGE = 0.0002
COMMISSION = 0.005          # $/share
RF_FALLBACK = 0.02          # ใช้เมื่อดึง ^IRX ไม่ได้
BLOCK = 12                  # block bootstrap (สัปดาห์)
N_BOOT = 10000
SEED = 42

GRID = dict(
    s=[20, 25, 30, 35, 40, 45, 50],
    w=[5, 10, 15, 20, 25],
    P_thr=[10, 15, 20, 25],
    P_mult=[1.0, 1.5, 2.0],
    B_mult=[1.0, 1.5, 2.0],
)

# ช่วง test ไม่ซ้อนกัน, train แบบ expanding
WF_WINDOWS = [
    ("2011-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("2011-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("2011-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("2011-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("2011-01-01", "2023-12-31", "2024-01-01", "2026-12-31"),
]

OLD_URL = ("https://raw.githubusercontent.com/hackingthemarkets/"
           "sentiment-fear-and-greed/master/datasets/fear-greed.csv")
CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start}"
CACHE = "fgi_cache.csv"

# CNN API ส่ง "ค่าเติมช่องว่าง" ช่วง 2020-09-19 -> 2020-12-31 (ไต่จาก ~0 แล้วล็อกที่ 50.0)
# จึงถือว่าช่วงนี้ไม่มีข้อมูล FGI ที่เชื่อถือได้
CNN_BAD_START = "2020-09-19"
CNN_TRUST_FROM = "2021-01-01"
ALL_PARTS = ("fear", "greed", "panic", "boost")   # ส่วนประกอบของกลยุทธ์ (ใช้ใน ablation)
PLATEAU_MIN_RUN = 5           # ค่าเดิมซ้ำ >= 5 วันทำการติดกัน = ค่าเติมช่องว่าง -> ตั้งเป็น NaN

# 1. DATA
def mask_plateaus(series, min_run=5):
    """ตั้งค่าที่ซ้ำกันติดต่อกัน >= min_run แถวเป็น NaN (สัญญาณของค่าเติมช่องว่าง)"""
    run_id = (series != series.shift()).cumsum()
    sizes = series.groupby(run_id).transform("size")
    return series.mask((sizes >= min_run) & series.notna())


def load_fgi():
    import requests
    parts = []
    if os.path.exists(CACHE):
        c = pd.read_csv(CACHE, parse_dates=["Date"]).set_index("Date")
        print(f"[FGI] cache       : {len(c)} rows, {c.index.min().date()} -> {c.index.max().date()}")
        parts.append(c[["FG"]])

    old = None
    try:
        old = pd.read_csv(OLD_URL, parse_dates=["Date"]).set_index("Date")
        old = old.rename(columns={"Fear Greed": "FG"})[["FG"]]
        old.index = old.index.normalize()
        print(f"[FGI] old source  : {len(old)} rows, {old.index.min().date()} -> {old.index.max().date()}")
        parts.append(old)
    except Exception as e:
        print(f"[FGI] old source FAILED: {e}")

    last_have = max([p.index.max() for p in parts]) if parts else None
    start = (last_have + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if last_have is not None else "2011-01-01"
    got_new = False
    try:
        headers = {
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            "Accept": "application/json",
            "Referer": "https://edition.cnn.com/markets/fear-and-greed",
        }
        resp = requests.get(CNN_URL.format(start=start), headers=headers, timeout=20)
        print(f"[FGI] CNN status  : {resp.status_code}")
        resp.raise_for_status()
        rows = resp.json()["fear_and_greed_historical"]["data"]
        new = pd.DataFrame(rows)
        new["Date"] = pd.to_datetime(new["x"], unit="ms").dt.normalize()   # normalize: ให้ตรงกับวันของ SPY
        new = new.set_index("Date").rename(columns={"y": "FG"})[["FG"]]
        print(f"[FGI] CNN source  : {len(new)} rows, {new.index.min().date()} -> {new.index.max().date()}")
        parts.append(new)
        got_new = True
    except Exception as e:
        print(f"[FGI] CNN source FAILED: {e}")

    if not parts:
        raise RuntimeError("ไม่มีข้อมูล FGI เลย (ทั้ง cache, old, CNN)")
    fg = pd.concat(parts).sort_index()
    fg = fg[~fg.index.duplicated(keep="last")]

    n0 = len(fg)
    fg = fg[(fg.index < CNN_BAD_START) | (fg.index >= CNN_TRUST_FROM)]
    print(f"[FGI] ตัดช่วงข้อมูลเสีย {CNN_BAD_START} -> {CNN_TRUST_FROM}: {n0 - len(fg)} แถว")
    fg = fg.copy()
    fg["FG"] = mask_plateaus(fg["FG"], PLATEAU_MIN_RUN)
    print(f"[FGI] ค่าเติมช่องว่าง (ซ้ำ >= {PLATEAU_MIN_RUN} วัน) ที่ถูกตั้งเป็น NaN: {int(fg['FG'].isna().sum())} แถว")

    if got_new:
        fg.reset_index().rename(columns={"index": "Date"}).to_csv(CACHE, index=False)
    return fg


def find_frozen_start(series, min_run=8):
    """คืนวันที่เริ่มต้นของช่วงที่ค่า FGI ซ้ำกันติดต่อกัน >= min_run สัปดาห์ (สัญญาณของ ffill ค้าง)"""
    v = series.values
    run_start = 0
    for i in range(1, len(v)):
        if v[i] != v[i - 1]:
            run_start = i
        elif i - run_start + 1 >= min_run:
            return series.index[run_start]
    return None


def build_weekly_data():
    import yfinance as yf
    fg = load_fgi()
    spy = yf.download("SPY", start=fg.index.min().strftime("%Y-%m-%d"),
                      auto_adjust=True, progress=False)
    px = spy["Close"].squeeze()
    px.index = pd.to_datetime(px.index).tz_localize(None)

    fg_al = fg["FG"].reindex(px.index).ffill(limit=5)      # ffill ได้ไม่เกิน 5 วันทำการ

    try:
        irx = yf.download("^IRX", start=px.index.min().strftime("%Y-%m-%d"),
                          auto_adjust=True, progress=False)["Close"].squeeze()
        irx.index = pd.to_datetime(irx.index).tz_localize(None)
        rf_ann = (irx / 100.0).reindex(px.index).ffill().bfill()
    except Exception as e:
        print(f"[RF] ^IRX FAILED ({e}) -> ใช้ค่าคงที่ {RF_FALLBACK:.0%}")
        rf_ann = pd.Series(RF_FALLBACK, index=px.index)

    data = pd.DataFrame({
        "Price": px.resample("W-FRI").last(),
        "FGI": fg_al.resample("W-FRI").last(),
        "RF": rf_ann.resample("W-FRI").last() / 52.0,       # อัตราดอกเบี้ยรายสัปดาห์
    }).dropna(subset=["Price", "RF"])          # FGI ที่เป็น NaN คงไว้ (ช่วงไม่มีข้อมูล)

    frozen = find_frozen_start(data["FGI"].dropna())
    if frozen is not None:
        print("\n" + "!" * 70)
        print(f"!! FGI ค้าง (ค่าเดิมซ้ำเกิน 8 สัปดาห์) ตั้งแต่ {frozen.date()}")
        print(f"!! ตัด backtest ให้เหลือก่อนวันนั้น ({len(data[data.index >= frozen])} สัปดาห์ถูกตัด)")
        print("!! ต้องแก้การดึงข้อมูล CNN ก่อน ถึงจะทดสอบช่วงหลังจากนี้ได้")
        print("!" * 70 + "\n")
        data = data[data.index < frozen]
    return data


def diagnose_fgi(data):
    g = data["FGI"].groupby(data.index.year)
    tbl = pd.DataFrame({"weeks": g.size(), "mean": g.mean().round(1),
                        "min": g.min().round(1), "max": g.max().round(1),
                        "unique": g.nunique(),
                        "missing": g.apply(lambda x: int(x.isna().sum()))})
    print("\n[FGI by year]  (unique ต่ำมาก = ข้อมูลผิดปกติ)")
    print(tbl.to_string())
    print(f"\nช่วงข้อมูลที่ใช้: {data.index.min().date()} -> {data.index.max().date()}  ({len(data)} สัปดาห์)")


# 2. STRATEGY + METRICS
def r_base(fgi, s, w, fear=True, greed=True):
    """น้ำหนักฐานตาม FGI  fear=เพิ่มเมื่อ FGI<s   greed=ลด/งดซื้อเมื่อ FGI>=s+w (Greed Lockout)
    หมายเหตุ: สูตรตรงกับโค้ดเดิม (คูณ 2) ให้เช็กว่าตรงกับสูตรในรายงานบทที่ 3
    และโซน taper ลดจาก 1 -> 0.8 แล้วกระโดดเป็น 0 ที่ s+w+10"""
    if fgi < s:
        return 1.0 + (s - fgi) / 100.0 * 2.0 if fear else 1.0
    if fgi < s + w:
        return 1.0
    if not greed:
        return 1.0
    if fgi < s + w + 10:
        return 1.0 - (fgi - (s + w)) / 100.0 * 2.0
    return 0.0


def run_backtest(price, fgi, rf, s=None, w=None, P_thr=None, P_mult=1.0, B_mult=1.0,
                 parts=ALL_PARTS, regular=False, constrained=True, baseline=BASELINE,
                 slippage=SLIPPAGE, commission=COMMISSION):
    """
    คืน ret = time-weighted weekly return (ตัดเงินที่เติมเข้าออกแล้ว)
    constrained=True  : ได้งบ baseline/สัปดาห์ เงินที่ไม่ใช้เก็บเป็นเงินสด (ได้ดอกเบี้ย rf) ไว้ใช้ต่อ
    constrained=False : ลงทุน baseline*R ได้เต็มที่ (เติมเงินเพิ่มจากภายนอกได้) ไม่มีเงินสดค้าง
    """
    n = len(price)
    shares, cash, v_prev, total_contrib = 0.0, 0.0, 0.0, 0.0
    last_fgi = np.nan
    ret = np.full(n, np.nan)
    value = np.zeros(n)
    Rs = np.zeros(n)
    cashv = np.zeros(n)
    for t in range(n):
        p = price[t]
        cash *= (1.0 + rf[t])
        v_pre = shares * p + cash
        if regular:
            R = 1.0
            invest = baseline
            flow_in = baseline
            total_contrib += baseline
        else:
            f = fgi[t]
            panic = P_mult if ("panic" in parts and f < P_thr) else 1.0
            rec = B_mult if ("boost" in parts and not np.isnan(last_fgi)
                             and last_fgi < P_thr and f >= P_thr) else 1.0
            R = (1.0 if np.isnan(f) else                     # ไม่มีสัญญาณ = DCA ปกติ
                 r_base(f, s, w, "fear" in parts, "greed" in parts) * panic * rec)
            if constrained:
                avail = cash + baseline
                invest = min(baseline * R, avail)
                cash = avail - invest
                flow_in = baseline
                total_contrib += baseline
            else:
                invest = baseline * R
                flow_in = invest
                total_contrib += invest
        new_sh = invest * (1.0 - slippage) / p
        new_sh -= new_sh * commission / p
        shares += max(new_sh, 0.0)
        v_end = shares * p + cash
        base_val = v_pre + flow_in
        if t > 0 and v_prev > 0 and base_val > 0:
            ret[t] = (v_pre / v_prev) * (v_end / base_val) - 1.0
        value[t] = v_end
        Rs[t] = R
        cashv[t] = cash
        v_prev = v_end
        last_fgi = fgi[t] if not regular else np.nan
    return {"ret": ret, "value": value, "R": Rs, "cash": cashv, "total_contrib": total_contrib}


def sharpe_of(r, f, periods=52):
    ex = r - f
    sd = ex.std(ddof=1)
    return ex.mean() * periods / (sd * np.sqrt(periods)) if sd > 0 else np.nan


def metrics(ret, rf, periods=52):
    m = ~np.isnan(ret)
    r, f = ret[m], rf[m]
    if len(r) < 10:
        return {k: np.nan for k in ["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar", "CVaR95"]}
    ex = r - f
    vol = r.std(ddof=1) * np.sqrt(periods)
    tdd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2)) * np.sqrt(periods)   # target downside deviation
    idx = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], idx]))[1:]
    mdd = (idx / peak - 1.0).min()
    years = len(r) / periods
    cagr = idx[-1] ** (1.0 / years) - 1.0
    q = np.quantile(r, 0.05)
    return {
        "CAGR": cagr, "Vol": vol, "Sharpe": sharpe_of(r, f, periods),
        "Sortino": ex.mean() * periods / tdd if tdd > 0 else np.nan,
        "MaxDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "CVaR95": r[r <= q].mean(),
    }

# 3. NESTED WALK-FORWARD
def grid_list():
    keys = list(GRID.keys())
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def arrays(d):
    return d["Price"].values, d["FGI"].values, d["RF"].values


def rank_params(train):
    px, fg, rf = arrays(train)
    rows = []
    for p in grid_list():
        out = run_backtest(px, fg, rf, **p)
        rows.append({**p, "Sharpe": metrics(out["ret"], rf)["Sharpe"]})
    return (pd.DataFrame(rows).dropna(subset=["Sharpe"])
            .sort_values("Sharpe", ascending=False).reset_index(drop=True))


def walk_forward(data):
    rows = []
    oos = {"dates": [], "rf": [], "Regular": [], "Best": [], "Avg10": []}
    last_best = None
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(WF_WINDOWS, 1):
        tr, te = data.loc[tr_s:tr_e], data.loc[te_s:te_e]
        if len(tr) < 104 or len(te) < 26:
            print(f"[WF] window {k} ข้าม (ข้อมูลไม่พอ: train={len(tr)}, test={len(te)})")
            continue
        ranked = rank_params(tr)
        best = {c: ranked.loc[0, c] for c in GRID}
        top = ranked.head(max(1, int(len(ranked) * 0.10)))
        avg = {c: float(top[c].mean()) for c in GRID}
        px, fg, rf = arrays(te)
        o_b = run_backtest(px, fg, rf, **best)
        o_a = run_backtest(px, fg, rf, **avg)
        o_r = run_backtest(px, fg, rf, regular=True)
        m_b, m_a, m_r = (metrics(o["ret"], rf) for o in (o_b, o_a, o_r))
        rows.append({
            "Window": k,
            "Test": f"{te.index.min():%Y-%m} -> {te.index.max():%Y-%m}",
            "Best params (chosen on train)": f"s={best['s']} w={best['w']} Pthr={best['P_thr']} "
                                             f"Pm={best['P_mult']} Bm={best['B_mult']}",
            "Train Sharpe": round(ranked.loc[0, "Sharpe"], 3),
            "Sharpe Regular": round(m_r["Sharpe"], 3),
            "Sharpe Best": round(m_b["Sharpe"], 3),
            "Sharpe Avg10": round(m_a["Sharpe"], 3),
            "D Best": round(m_b["Sharpe"] - m_r["Sharpe"], 3),
            "D Avg10": round(m_a["Sharpe"] - m_r["Sharpe"], 3),
        })
        keep = ~np.isnan(o_b["ret"]) & ~np.isnan(o_r["ret"])
        oos["dates"].append(te.index[keep])
        oos["rf"].append(rf[keep])
        oos["Regular"].append(o_r["ret"][keep])
        oos["Best"].append(o_b["ret"][keep])
        oos["Avg10"].append(o_a["ret"][keep])
        last_best = best
        print(f"[WF] window {k} done  test {te.index.min().date()} -> {te.index.max().date()}")
    if not rows:
        raise RuntimeError("ไม่มี window ใดที่มีข้อมูลพอ")
    for key in ("rf", "Regular", "Best", "Avg10"):
        oos[key] = np.concatenate(oos[key])
    oos["dates"] = pd.DatetimeIndex(np.concatenate([d.values for d in oos["dates"]]))
    return pd.DataFrame(rows), oos, last_best


# 4. STATISTICS
def block_bootstrap_delta_sharpe(ra, rb, rf, block=BLOCK, B=N_BOOT, seed=SEED):
    n = len(ra)
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]

    def sh(X):
        ex = X - rf[idx]
        return ex.mean(1) * 52 / (ex.std(1, ddof=1) * np.sqrt(52))

    boot = sh(ra[idx]) - sh(rb[idx])
    obs = sharpe_of(ra, rf) - sharpe_of(rb, rf)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = float(np.mean(np.abs(boot - boot.mean()) >= abs(obs)))
    return {"obs": obs, "lo": lo, "hi": hi, "p": p, "boot": boot}


def paired_tests(ra, rb):
    from scipy import stats
    d = ra - rb
    t = stats.ttest_rel(ra, rb)
    try:
        wl = stats.wilcoxon(d)
        wp = wl.pvalue
    except Exception:
        wp = np.nan
    return {"t_p": float(t.pvalue), "wilcoxon_p": float(wp),
            "cohen_d": float(d.mean() / d.std(ddof=1)), "mean_diff_weekly": float(d.mean())}


def constraint_comparison(data, params):
    """วัตถุประสงค์ข้อ 2: จำกัดเงิน vs ไม่จำกัดเงิน (พารามิเตอร์เดียวกัน, ทั้งช่วงข้อมูล = in-sample)"""
    px, fg, rf = arrays(data)
    rows = []
    for name, kw in [("Regular DCA", dict(regular=True)),
                     ("FGI-DCA constrained", dict(constrained=True, **params)),
                     ("FGI-DCA unconstrained", dict(constrained=False, **params))]:
        o = run_backtest(px, fg, rf, **kw)
        m = metrics(o["ret"], rf)
        rows.append({"Strategy": name, "Sharpe": round(m["Sharpe"], 3),
                     "CAGR(TWR) %": round(m["CAGR"] * 100, 2), "MaxDD %": round(m["MaxDD"] * 100, 2),
                     "Total invested $": round(o["total_contrib"]),
                     "Final value $": round(o["value"][-1]),
                     "Final / invested": round(o["value"][-1] / o["total_contrib"], 3)})
    return pd.DataFrame(rows)


# 5. PLOTS
def make_plots(data, oos, wf_df, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    a = ax[0, 0]
    a.plot(data.index, data["FGI"], lw=0.8, color="#2E86AB")
    a.set_title("Weekly FGI used in the backtest (must keep moving to the end)")
    a.set_ylabel("FGI")

    a = ax[0, 1]
    for name, col in [("Regular", "#F18F01"), ("Best", "#2E86AB"), ("Avg10", "#06A77D")]:
        a.plot(oos["dates"], np.cumprod(1 + oos[name]), label=name, color=col, lw=2)
    a.set_yscale("log")
    a.set_title("Stitched out-of-sample growth of 1 unit (time-weighted)")
    a.legend()

    a = ax[1, 0]
    for key, lab in (("Best", "Best - Regular"), ("Avg10", "Avg10 - Regular")):
        bb = res[key]["boot"]
        bb = bb[np.isfinite(bb)]
        if len(bb):
            a.hist(bb, bins=80, alpha=0.6, density=True, label=lab)
    a.axvline(0, color="red")
    a.set_title(f"Block bootstrap (block={BLOCK}w) of Delta Sharpe")
    a.legend()

    a = ax[1, 1]
    x = np.arange(len(wf_df))
    a.bar(x - 0.25, wf_df["Sharpe Regular"], 0.25, label="Regular", color="#F18F01")
    a.bar(x, wf_df["Sharpe Best"], 0.25, label="Best", color="#2E86AB")
    a.bar(x + 0.25, wf_df["Sharpe Avg10"], 0.25, label="Avg10", color="#06A77D")
    a.set_xticks(x)
    a.set_xticklabels(wf_df["Test"], rotation=20, fontsize=8)
    a.set_title("Sharpe per out-of-sample window")
    a.legend()

    plt.tight_layout()
    plt.savefig("fgi_dca_v2_results.png", dpi=200)
    print("saved fgi_dca_v2_results.png")


# 5b. ABLATION  (พารามิเตอร์กำหนดล่วงหน้าจากโซนของ CNN ไม่ optimize)

# CNN: Extreme Fear < 25, Fear 25-44, Neutral 45-55, Greed 56-75, Extreme Greed > 75
PRESETS = {
    "A (lockout>=75, x1.5)": dict(s=45, w=20, P_thr=25, P_mult=1.5, B_mult=1.5),   # หลัก
    "B (lockout>=75, x2.0)": dict(s=45, w=20, P_thr=25, P_mult=2.0, B_mult=2.0),   # ทดสอบความไว: ตัวคูณ
    "C (lockout>=65, x1.5)": dict(s=45, w=10, P_thr=25, P_mult=1.5, B_mult=1.5),   # ทดสอบความไว: เกณฑ์งดซื้อ
}
VARIANTS = [
    ("Lockout only",          ("greed",)),
    ("Full strategy",         ("fear", "greed", "panic", "boost")),
    ("Full - Panic Buy",      ("fear", "greed", "boost")),
    ("Full - Boost Recovery", ("fear", "greed", "panic")),
    ("Full - Fear weighting", ("greed", "panic", "boost")),
    ("Full - Lockout",        ("fear", "panic", "boost")),
]


def ablation_one(data, params):
    px, fg, rf = arrays(data)
    reg = run_backtest(px, fg, rf, regular=True)

    def row_for(name, out, is_reg=False):
        m = metrics(out["ret"], rf)
        row = {"Variant": name, "Sharpe": round(m["Sharpe"], 3),
               "CAGR %": round(m["CAGR"] * 100, 2), "MaxDD %": round(m["MaxDD"] * 100, 2),
               "Final/invested": round(out["value"][-1] / out["total_contrib"], 3),
               "Avg cash %": round(100 * float(np.mean(out["cash"] / out["value"])), 1),
               "Weeks R=0 %": round(100 * float(np.mean(out["R"] == 0)), 1)}
        if is_reg:
            row.update({"dSharpe": 0.0, "CI low": np.nan, "CI high": np.nan, "boot p": np.nan})
        else:
            keep = ~np.isnan(out["ret"]) & ~np.isnan(reg["ret"])
            b = block_bootstrap_delta_sharpe(out["ret"][keep], reg["ret"][keep], rf[keep])
            row.update({"dSharpe": round(b["obs"], 4), "CI low": round(b["lo"], 4),
                        "CI high": round(b["hi"], 4), "boot p": round(b["p"], 3)})
        return row

    rows = [row_for("Regular DCA", reg, is_reg=True)]
    series = {"Regular DCA": reg["ret"]}
    for name, parts in VARIANTS:
        out = run_backtest(px, fg, rf, parts=parts, **params)
        rows.append(row_for(name, out))
        series[name] = out["ret"]
    return pd.DataFrame(rows), series


def plot_ablation(data, df, series, preset_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for name, col in [("Regular DCA", "#F18F01"), ("Lockout only", "#E63946"), ("Full strategy", "#2E86AB")]:
        r = np.nan_to_num(series[name], nan=0.0)
        ax[0].plot(data.index, np.cumprod(1 + r), label=name, color=col, lw=2)
    ax[0].set_yscale("log")
    ax[0].set_title(f"Growth of 1 unit (time-weighted), preset {preset_name}")
    ax[0].legend()
    d = df[df["Variant"] != "Regular DCA"].reset_index(drop=True)
    y = np.arange(len(d))
    ax[1].errorbar(d["dSharpe"], y, xerr=[d["dSharpe"] - d["CI low"], d["CI high"] - d["dSharpe"]],
                   fmt="o", color="#2E86AB", capsize=5)
    ax[1].axvline(0, color="red", ls="--")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(d["Variant"])
    ax[1].set_title("Delta Sharpe vs Regular DCA (95% block-bootstrap CI)")
    plt.tight_layout()
    plt.savefig("fgi_dca_ablation.png", dpi=200)
    print("saved fgi_dca_ablation.png")


def ablation_report(data):
    print("\n" + "=" * 70)
    print("ABLATION  (พารามิเตอร์กำหนดล่วงหน้า ไม่ optimize, ทั้งช่วงข้อมูลแบบจำกัดเงิน, block bootstrap)")
    print("=" * 70)
    tables, first = [], None
    for pname, params in PRESETS.items():
        df, series = ablation_one(data, params)
        print(f"\n--- Preset {pname}   {params} ---")
        print(df.to_string(index=False))
        t = df.copy()
        t.insert(0, "Preset", pname)
        tables.append(t)
        if first is None:
            first = (df, series, pname)
    pd.concat(tables).to_csv("ablation_results.csv", index=False)
    print("\nวิธีอ่าน:")
    print(" - 'Full - Lockout' ต้องเท่ากับ Regular DCA (แบบจำกัดเงินไม่มีเงินสดให้ซื้อเพิ่ม ถ้าไม่งดซื้อก่อน)")
    print(" - 'Lockout only' = ผลของการถือเงินสดอย่างเดียว (ไม่เคยซื้อเพิ่มคืน)")
    print(" - ส่วนที่ Panic/Boost/Fear weighting เพิ่มให้ ดูจาก Full เทียบกับ Full - ส่วนนั้น")
    print(f" - ทดสอบ 6 แบบต่อ preset: ถ้าอ้างนัยสำคัญให้ใช้เกณฑ์เข้มขึ้น (p < 0.05/6 ~ 0.008) หรือดู CI เป็นหลัก")
    plot_ablation(data, *first)


# 6. MAIN
def report(data):
    wf_df, oos, last_best = walk_forward(data)
    print("\n" + "=" * 70)
    print("NESTED WALK-FORWARD (พารามิเตอร์เลือกจาก train เท่านั้น, test ไม่ซ้อนกัน)")
    print("=" * 70)
    print(wf_df.to_string(index=False))
    wf_df.to_csv("wf_windows_v2.csv", index=False)

    print("\n" + "=" * 70)
    print("STITCHED OUT-OF-SAMPLE METRICS")
    print("=" * 70)
    rows = []
    for name in ("Regular", "Best", "Avg10"):
        m = metrics(oos[name], oos["rf"])
        rows.append({"Strategy": name, **{k: round(v, 4) for k, v in m.items()}})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"OOS weeks: {len(oos['rf'])}  ({oos['dates'].min().date()} -> {oos['dates'].max().date()})")

    print("\n" + "=" * 70)
    print(f"BLOCK BOOTSTRAP (block={BLOCK}, B={N_BOOT}) + PAIRED TESTS   [Delta = FGI - Regular]")
    print("=" * 70)
    res = {}
    for name in ("Best", "Avg10"):
        b = block_bootstrap_delta_sharpe(oos[name], oos["Regular"], oos["rf"])
        pt = paired_tests(oos[name], oos["Regular"])
        res[name] = b
        print(f"{name:6s} dSharpe={b['obs']:+.4f}  95%CI=[{b['lo']:+.4f},{b['hi']:+.4f}]  "
              f"boot p={b['p']:.3f}  | paired-t p={pt['t_p']:.3f}  wilcoxon p={pt['wilcoxon_p']:.3f}  "
              f"cohen_d={pt['cohen_d']:+.3f}")
    print("(paired-t/Wilcoxon เทียบ 'ผลตอบแทนเฉลี่ยรายสัปดาห์' ไม่ใช่ Sharpe และข้อมูลมี autocorrelation"
          " ให้ถือ bootstrap เป็นหลัก)")

    print("\n" + "=" * 70)
    print("CONSTRAINED vs UNCONSTRAINED  (พารามิเตอร์ของ window สุดท้าย, ทั้งช่วงข้อมูล = in-sample)")
    print("=" * 70)
    cc = constraint_comparison(data, last_best)
    print(cc.to_string(index=False))
    print("ระวัง: unconstrained ลงเงินรวมไม่เท่ากัน จึงเทียบ Sharpe/TWR และ Final/invested ไม่ใช่มูลค่าสุดท้ายตรง ๆ")
    cc.to_csv("constraint_comparison_v2.csv", index=False)

    make_plots(data, oos, wf_df, res)


def main():
    import sys
    data = build_weekly_data()
    diagnose_fgi(data)
    if "ablation" not in sys.argv[1:]:      # python fgi_dca_v2.py ablation = รันเฉพาะ ablation
        report(data)
    ablation_report(data)


if __name__ == "__main__":
    main()
