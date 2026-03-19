"""V4 Quantum Portfolio Dashboard - Streamlit 1.55+"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    _HAS_MATPLOTLIB = False

# -- Paths ------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "output", "v3")
CACHE_DIR  = os.path.join(_HERE, "cache", "v3")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

sys.path.insert(0, os.path.join(_HERE, "src"))
_ROOT_SRC = os.path.join(os.path.dirname(_HERE), "src")
if os.path.isdir(_ROOT_SRC) and _ROOT_SRC not in sys.path:
    sys.path.insert(0, _ROOT_SRC)
_V4_SRC = os.path.join(_HERE, "v4_ultimate", "src")
if os.path.isdir(_V4_SRC) and _V4_SRC not in sys.path:
    sys.path.insert(0, _V4_SRC)

# -- Optional imports -------------------------------------------------------
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    from v4_interactive_engine import fast_mc, format_aum
    _HAS_ENGINE = True
except ImportError:
    _HAS_ENGINE = False
    def format_aum(aum):
        return f"${aum/1e6:.1f}M" if aum < 1e9 else f"${aum/1e9:.2f}B"

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#13151f",
    font=dict(family="Inter, Arial, sans-serif", color="#e0e0e0", size=12),
    margin=dict(l=10, r=10, t=48, b=10),
)

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="V4 Quantum Portfolio",
    page_icon="\u269b\ufe0f",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    "<style>"
    "[data-testid='metric-container']{padding:8px 12px}"
    "h4{color:#4a9eff;margin-top:1.2rem}"
    "</style>",
    unsafe_allow_html=True,
)

# -- Data loaders -----------------------------------------------------------

@st.cache_data(ttl=3600)
def load_regime():
    from regime_detector import RegimeDetector
    rd = RegimeDetector(cache_dir=CACHE_DIR)
    regime = rd.detect(verbose=False)
    return regime, rd.get_components(), rd.get_basis_weights()


@st.cache_data(ttl=300)
def load_portfolio():
    if not os.path.isdir(OUTPUT_DIR):
        return None, None
    files = sorted(
        [f for f in os.listdir(OUTPUT_DIR)
         if f.startswith("ultimate_portfolio_") and f.endswith(".csv")],
        reverse=True,
    )
    if not files:
        return None, None
    df = pd.read_csv(os.path.join(OUTPUT_DIR, files[0]))
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    date_str = files[0].replace("ultimate_portfolio_", "").replace(".csv", "")
    return df, date_str


@st.cache_data(ttl=300)
def load_cal_meta():
    p = os.path.join(OUTPUT_DIR, "calibration_meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_ibkr_orders():
    p = os.path.join(OUTPUT_DIR, "ibkr_orders.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def show_img(filename, caption="", src_dir=None):
    d = src_dir or OUTPUT_DIR
    p = os.path.join(d, filename)
    if os.path.exists(p):
        st.image(p, caption=caption, width="stretch")
    else:
        st.caption(f"*Not generated yet:* `{filename}`")


def render_sector_bar(portfolio):
    sec = (portfolio.groupby("sector")["weight"]
           .sum().mul(100).sort_values(ascending=False))
    if sec.empty:
        st.info("No sector allocation data available.")
        return

    labels = list(sec.index[::-1])
    values = sec.values[::-1]

    if _HAS_PLOTLY:
        palette = [
            "#4a9eff",
            "#72d572",
            "#ffb347",
            "#ff6b9d",
            "#40e0d0",
            "#f9ca24",
            "#c3a6ff",
            "#7ec8e3",
        ]
        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(
                    color=[palette[i % len(palette)] for i in range(len(values))],
                    line=dict(color="rgba(255,255,255,0.25)", width=1),
                ),
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            )
        )
        fig.update_layout(
            **_DARK,
            title=dict(text="Sector Weights", x=0.02, xanchor="left"),
            xaxis_title="Allocation (%)",
            yaxis_title=None,
            height=max(280, 44 * len(values) + 80),
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
        return

    if _HAS_MATPLOTLIB:
        colors = plt.cm.tab20.colors
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(
            labels,
            values,
            color=[colors[i % len(colors)] for i in range(len(values))],
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_xlabel("Allocation (%)")
        ax.set_title("Sector Weights", fontsize=10)
        ax.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        return

    st.dataframe(
        sec.rename("allocation_pct").round(2).rename_axis("sector").reset_index(),
        width="stretch",
        hide_index=True,
    )

# -- HEADER -----------------------------------------------------------------
hl, hr = st.columns([4, 1])
with hl:
    st.title("Quantum Portfolio - V4 Ultimate")
    st.caption("Born-rule selection - Regime-adaptive weights - Empirical phase calibration")
with hr:
    st.caption(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    if st.button("Refresh"):
        load_regime.clear()
        st.rerun()

# -- Regime strip -----------------------------------------------------------
try:
    regime, components, basis_weights = load_regime()
    _icon = {"BULL": "BULL", "NEUTRAL": "NEUTRAL", "BEAR": "BEAR"}.get(regime, "?")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Market Regime", f"{_icon}  {regime}",
              delta=f"score {components.get('score', 0):+.2f}")
    vix = components.get("vix", 0) or 0
    r2.metric("VIX", f"{vix:.1f}",
              delta="High" if components.get("bear_signals", {}).get("vix_high") else "Normal",
              delta_color="inverse")
    yc = components.get("yield_curve")
    r3.metric("10Y-2Y Spread", f"{yc:+.2f}%" if yc is not None else "N/A",
              delta="Inverted" if components.get("bear_signals", {}).get("curve_inv") else "Normal",
              delta_color="inverse")
    cs = components.get("credit_spread")
    r4.metric("HY Credit Spread", f"{cs:.2f}%" if cs is not None else "N/A",
              delta="Wide" if components.get("bear_signals", {}).get("spread_wide") else "Normal",
              delta_color="inverse")
    spy_px = components.get("spy_price", 0) or 0
    spy_sma = components.get("spy_sma200", 1) or 1
    r5.metric("SPY vs 200-SMA", f"${spy_px:.2f}",
              delta=f"{(spy_px / spy_sma - 1) * 100:+.1f}%",
              delta_color="normal")
    _regime_ok = True
except Exception as _e:
    st.warning(f"Regime data unavailable: {_e}")
    regime, components, basis_weights = "N/A", {}, {}
    _regime_ok = False

st.divider()

# -- TABS -------------------------------------------------------------------
tab_overview, tab_holdings, tab_metrics, tab_mc, tab_dd, tab_engine, tab_q3d = st.tabs([
    "Overview",
    "Holdings",
    "Performance",
    "Monte Carlo",
    "Drawdown",
    "Quant Engine",
    "Quantum 3D",
])

# -- TAB 1: OVERVIEW -------------------------------------------------------
with tab_overview:
    cal = load_cal_meta()
    portfolio, port_date = load_portfolio()

    if portfolio is not None and cal:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Holdings", len(portfolio))
        s2.metric("Portfolio date", port_date or "N/A")
        best_a = cal.get("best_a", "?")
        best_b = cal.get("best_b", "?")
        s3.metric("Phase a", f"{best_a:.3f}" if isinstance(best_a, float) else str(best_a))
        s4.metric("Phase b", f"{best_b:.3f}" if isinstance(best_b, float) else str(best_b))
        oos = cal.get("oos_valid")
        s5.metric("OOS Validity", "ACCEPTED" if oos else ("CAUTION" if oos is False else "N/A"))
        st.divider()

    st.markdown("## How This Strategy Works")
    ca, cb = st.columns(2)

    with ca:
        st.markdown("""
### 1. Universe & Data Pipeline
Full US equity universe from **Finviz** (500+ tickers, market cap >= $1B).
**3 years of daily prices** via yFinance. Macro conditions (VIX, yield curve,
HY spreads, SPY vs 200-SMA) from **FRED** classify the regime (Bull/Neutral/Bear).

---

### 2. Risk-Adjusted Scoring

| Pillar | Metric | Measures |
|---|---|---|
| **Risk** | Sortino | Return per downside vol (RF 4%) |
| **Consistency** | Win Rate | % positive months over 3Y |
| **Trend** | Momentum | 12-month decay-weighted momentum |
| **Tail** | Max Drawdown | Worst peak-to-trough in 3Y |

Blended into a **composite rank**. Sortino < 0.5 or drawdown > -25% filtered.

---

### 3. Quantum Selection (Born Rule)
> **theta(phi) = a * pi * (1 - phi)^b**
> **|psi|^2 = cos^2(theta)**

Parameters **a**, **b** grid-searched via walk-forward OOS (k=5 folds)
maximising out-of-sample Sharpe.
""")

    with cb:
        st.markdown("""
### 4. Regime-Adaptive Basis Weights

| Regime | Emphasis |
|---|---|
| Bull | Higher momentum |
| Neutral | Balanced blend |
| Bear | Sortino + drawdown protection |

L2-regularised + **Q-table RL** blend (70% L2 + 30% RL).

---

### 5. Cointegration Rebalancing
Each holding vs sector ETF. Rolling Z-score:
- **Z > +2**: TRIM (overperforming)
- **Z < -2**: ROTATE OUT (lagging)

---

### 6. Risk Controls
- Bootstrap MC: 1000 paths x 252 days (21d block resample)
- Max drawdown cap: -25%
- Min Sortino: 0.5
- Sector cap: 35%

---

### 7. IBKR Export
JSON order book for TWS paper trading.

*Run* `StockPickerV4_Ultimate.ipynb` *then* `PortfolioReport_V4.ipynb`
""")

    if basis_weights and _regime_ok:
        st.divider()
        st.markdown("#### Current Regime Basis Weights")
        bw_df = pd.DataFrame([basis_weights]).rename(index={0: "weight"})
        st.dataframe(bw_df.style.format("{:.0%}"), width="stretch")

# -- TAB 2: HOLDINGS -------------------------------------------------------
with tab_holdings:
    portfolio, port_date = load_portfolio()

    if portfolio is None:
        st.warning("No portfolio CSV found. Run StockPickerV4_Ultimate.ipynb first.")
    else:
        st.caption(f"Portfolio as of **{port_date}** -- {len(portfolio)} holdings")
        ct, cc = st.columns([3, 2])

        with ct:
            d = portfolio.copy()
            if "weight" in d.columns:
                d["wt %"] = (d["weight"] * 100).round(2)
            if "psi_sq" in d.columns:
                d["|psi|^2"] = d["psi_sq"].round(4)
            if "sortino" in d.columns:
                d["sortino"] = d["sortino"].round(2)
            if "win_rate" in d.columns:
                d["win_rate %"] = (d["win_rate"] * 100).round(1)
            if "zscore" in d.columns:
                d["Z-score"] = d["zscore"].round(2)
            keep = [c for c in
                    ["ticker", "sector", "wt %", "|psi|^2", "sortino",
                     "win_rate %", "momentum", "max_dd", "Z-score"]
                    if c in d.columns]
            st.dataframe(
                d[keep].sort_values("wt %", ascending=False).reset_index(drop=True),
                width="stretch", height=540,
            )

        with cc:
            if "sector" in portfolio.columns and "weight" in portfolio.columns:
                render_sector_bar(portfolio)

        if "zscore" in portfolio.columns:
            flagged = (portfolio[portfolio["zscore"].abs() >= 2.0]
                       .sort_values("zscore", key=abs, ascending=False))
            n = len(flagged)
            label = (f"{n} Rebalancing Signal{'s' if n != 1 else ''}"
                     if n else "No Rebalancing Signals")
            with st.expander(label, expanded=n > 0):
                if flagged.empty:
                    st.info("All holdings within 2-sigma of sector ETF.")
                else:
                    fd = flagged[[c for c in ["ticker", "sector", "weight", "zscore", "psi_sq"]
                                  if c in flagged.columns]].copy()
                    if "weight" in fd.columns:
                        fd["wt %"] = (fd["weight"] * 100).round(2)
                        fd = fd.drop(columns=["weight"])
                    fd["action"] = flagged["zscore"].apply(
                        lambda z: "TRIM" if z > 2 else "ROTATE OUT")
                    st.dataframe(fd, width="stretch")

        st.divider()
        show_img("v4_sector_comparison.png", "V4 Quantum vs equal-weight sector allocation")

# -- TAB 3: PERFORMANCE ----------------------------------------------------
with tab_metrics:
    cal = load_cal_meta()
    perf = cal.get("performance", {})

    if perf:
        st.markdown("#### Risk-Adjusted Return Summary")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        def _f(v, fmt):
            return (fmt % v) if isinstance(v, (int, float)) else "N/A"
        m1.metric("Sharpe",    _f(perf.get("sharpe"),     "%.2f"))
        m2.metric("Sortino",   _f(perf.get("sortino"),    "%.2f"))
        m3.metric("Calmar",    _f(perf.get("calmar"),     "%.2f"))
        m4.metric("Ann. Ret",  _f(perf.get("ann_return"), "%.1f%%"))
        m5.metric("Ann. Vol",  _f(perf.get("ann_vol"),    "%.1f%%"))
        m6.metric("Win Rate",  _f(perf.get("win_rate"),   "%.0f%%"))
        st.divider()

    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Portfolio vs S&P 500")
        show_img("v4_vs_spy.png", "4-panel comparison vs SPY", src_dir=REPORT_DIR)
    with cr:
        st.markdown("#### Risk Metrics Breakdown")
        show_img("v4_risk_metrics.png", "Sharpe / Sortino / VaR / Calmar", src_dir=REPORT_DIR)

    st.divider()
    st.markdown("#### Top 20 Holdings Scorecard")
    show_img("v4_top_holdings.png", "Full V4 scorecard", src_dir=REPORT_DIR)
    st.markdown("#### Sector Allocation")
    show_img("v4_sector_allocation.png", "Sector breakdown", src_dir=REPORT_DIR)

# -- TAB 4: MONTE CARLO ----------------------------------------------------
with tab_mc:
    cal = load_cal_meta()
    mc_stats = cal.get("monte_carlo", {})

    if mc_stats:
        def _pct(v):
            return f"{v:.1%}" if isinstance(v, float) else "N/A"
        ms1, ms2, ms3, ms4, ms5 = st.columns(5)
        ms1.metric("V4 Median 1Y",       _pct(mc_stats.get("quantum_median_return")))
        ms2.metric("Classical Median 1Y", _pct(mc_stats.get("classical_median_return")))
        ms3.metric("SPY Median 1Y",       _pct(mc_stats.get("spy_median_return")))
        ms4.metric("VaR 95%",  _pct(mc_stats.get("var_95")),  delta_color="inverse")
        ms5.metric("CVaR 95%", _pct(mc_stats.get("cvar_95")), delta_color="inverse")
        st.divider()

    _mc_port, _ = load_portfolio()

    if _HAS_PLOTLY and _HAS_ENGINE and _mc_port is not None:
        st.markdown("#### Live Monte Carlo Simulation")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _aum_def = float(cal.get("aum", 10_000_000)) if cal else 10_000_000
            _aum_m = st.number_input("AUM ($M)", 1.0, 500.0,
                                     round(_aum_def / 1e6, 1), 1.0, key="mc_aum")
            _aum = _aum_m * 1e6
        with c2:
            _nsims = st.selectbox("Simulations", [500, 1000, 2000, 5000], index=1, key="mc_n")
        with c3:
            _hor_lbl = st.selectbox("Horizon", ["1Y", "2Y", "3Y", "5Y"], key="mc_h")
            _hor_days = {"1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260}[_hor_lbl]
        with c4:
            st.markdown("&nbsp;", unsafe_allow_html=True)
            _run = st.button("Run MC", type="primary", key="mc_go")

        if _run or st.session_state.get("_mc_res") is not None:
            if _run:
                with st.spinner(f"Running {_nsims:,} paths x {_hor_lbl}..."):
                    try:
                        res = fast_mc(_mc_port.to_json(), _nsims, _hor_days, _aum)
                        st.session_state["_mc_res"] = res
                        st.session_state["_mc_aum"] = _aum
                        st.session_state["_mc_hor"] = _hor_lbl
                    except Exception as e:
                        st.error(f"MC error: {e}")
                        st.session_state["_mc_res"] = None

            res = st.session_state.get("_mc_res")
            if res is not None:
                mc_aum = st.session_state.get("_mc_aum", _aum)
                mc_hor = st.session_state.get("_mc_hor", _hor_lbl)
                s = res["stats"]
                x = res["x_years"]
                paths = res["paths"]

                l1, l2, l3, l4, l5, l6 = st.columns(6)
                l1.metric("Median CAGR",   f"{s['ann_cagr']:+.1f}%")
                l2.metric("Sharpe",        f"{s['sharpe']:.2f}")
                l3.metric("VaR 95%",       f"{s['var95']:.1f}%",  delta_color="inverse")
                l4.metric("CVaR 95%",      f"{s['cvar95']:.1f}%", delta_color="inverse")
                l5.metric("Median Max DD", f"{s['mdd_median']:.1f}%", delta_color="inverse")
                l6.metric("Win Rate",      f"{s['win_rate']:.1f}%")
                st.divider()

                # Fan chart
                st.markdown(f"#### Fan Chart - {format_aum(mc_aum)} - {mc_hor}")
                p5  = np.array(paths["p5"])  / mc_aum
                p25 = np.array(paths["p25"]) / mc_aum
                p50 = np.array(paths["p50"]) / mc_aum
                p75 = np.array(paths["p75"]) / mc_aum
                p95 = np.array(paths["p95"]) / mc_aum
                GOLD = "#F4C430"

                fig_fan = go.Figure()
                fig_fan.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(width=0),
                                             showlegend=False))
                fig_fan.add_trace(go.Scatter(x=x, y=p5, fill="tonexty",
                                             fillcolor="rgba(244,196,48,0.09)",
                                             line=dict(width=0), name="5-95th pctile"))
                fig_fan.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0),
                                             showlegend=False))
                fig_fan.add_trace(go.Scatter(x=x, y=p25, fill="tonexty",
                                             fillcolor="rgba(244,196,48,0.20)",
                                             line=dict(width=0), name="25-75th pctile"))
                fig_fan.add_trace(go.Scatter(x=x, y=p50, mode="lines",
                                             line=dict(color=GOLD, width=2.5), name="Median"))
                fig_fan.add_hline(y=1.0, line_dash="dash", line_color="#555",
                                  annotation_text=f"Entry ({format_aum(mc_aum)})",
                                  annotation_position="right")
                fig_fan.add_hline(y=1 + s["var95"] / 100, line_dash="dot",
                                  line_color="#ff4b4b",
                                  annotation_text=f"VaR 95%: {s['var95']:.1f}%",
                                  annotation_position="right")
                fig_fan.update_layout(**_DARK, height=420,
                                      xaxis_title=f"Years ({mc_hor})",
                                      yaxis_title="Portfolio value (x entry)",
                                      legend=dict(orientation="h", y=1.02, x=0),
                                      hovermode="x unified")
                st.plotly_chart(fig_fan, width="stretch")
                st.download_button("Download fan chart HTML",
                                   data=fig_fan.to_html(include_plotlyjs="cdn").encode(),
                                   file_name="mc_fan_chart.html", mime="text/html",
                                   key="mc_dl_fan")
                st.divider()

                # Return histogram
                st.markdown("#### 1-Year Return Distribution")
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=res["final_rets"], nbinsx=60,
                    marker_color=GOLD, opacity=0.75, name="Returns",
                    hovertemplate="Return: %{x:.1f}%<br>Count: %{y}<extra></extra>"))
                for v, lbl, col in [
                    (s["var95"],  f"VaR 95%: {s['var95']:.1f}%",  "#ff4b4b"),
                    (s["cvar95"], f"CVaR 95%: {s['cvar95']:.1f}%", "#ff7a50"),
                ]:
                    fig_hist.add_vline(x=v, line_dash="dash", line_color=col,
                                       annotation_text=lbl, annotation_font_color=col,
                                       annotation_position="top right")
                fig_hist.add_vline(x=0, line_color="#555", line_width=1)
                fig_hist.update_layout(**_DARK, height=320,
                                       xaxis_title="1-Year Return (%)",
                                       yaxis_title="Count", showlegend=False)
                st.plotly_chart(fig_hist, width="stretch")
        else:
            st.info("Set AUM / simulations above then click **Run MC**.")

    elif not _HAS_PLOTLY:
        st.warning("plotly not installed. pip install plotly then restart.")
    elif _mc_port is None:
        st.warning("No portfolio found -- run the notebook first.")

    st.divider()
    show_img("v4_mc_fan_chart.png", "Notebook MC fan chart -- V4 vs Classical vs SPY")
    cl2, cr2 = st.columns(2)
    with cl2:
        st.markdown("#### Return Distribution")
        show_img("v4_mc_return_dist.png", "Return distribution with VaR/CVaR")
    with cr2:
        st.markdown("#### Max Drawdown Distribution")
        show_img("v4_mc_drawdown_dist.png", "MC max drawdown -- V4 vs Classical vs SPY")

# -- TAB 5: DRAWDOWN -------------------------------------------------------
with tab_dd:
    cal = load_cal_meta()
    dd_mc = cal.get("drawdown", {})
    dd_hist = cal.get("drawdown_hist", {})

    if dd_mc or dd_hist:
        st.markdown("#### Drawdown Summary")
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("V4 Median Max DD (MC)",
                  f"{dd_mc.get('quantum_median_mdd', 'N/A')}%" if dd_mc else "N/A",
                  delta_color="inverse")
        d2.metric("Classical Median Max DD",
                  f"{dd_mc.get('classical_median_mdd', 'N/A')}%" if dd_mc else "N/A",
                  delta_color="inverse")
        d3.metric("SPY Median Max DD",
                  f"{dd_mc.get('spy_median_mdd', 'N/A')}%" if dd_mc else "N/A",
                  delta_color="inverse")
        d4.metric("Current Drawdown",
                  f"{dd_hist.get('current_dd', 'N/A')}%" if dd_hist else "N/A",
                  delta_color="inverse")
        d5.metric("1Y Max Drawdown",
                  f"{dd_hist.get('max_dd_1y', 'N/A')}%" if dd_hist else "N/A",
                  delta=f"avg {dd_hist.get('avg_dd_1y', 0):.1f}%" if dd_hist else None,
                  delta_color="inverse")
        st.divider()

    show_img("v4_drawdown_report.png", "Composite drawdown risk report")
    cuw, chm = st.columns(2)
    with cuw:
        st.markdown("#### Underwater Equity Curve (1Y)")
        show_img("v4_drawdown_underwater.png", "Underwater chart vs SPY")
    with chm:
        st.markdown("#### Monthly Return Calendar")
        show_img("v4_drawdown_monthly_heatmap.png", "Monthly return heatmap")

# -- TAB 6: QUANT ENGINE ---------------------------------------------------
with tab_engine:
    cal = load_cal_meta()

    st.markdown("### Phase Calibration & OOS Validation")
    if not cal:
        st.warning("No calibration_meta.json -- run the notebook first.")
    else:
        best_a = cal.get("best_a", "?")
        best_b = cal.get("best_b", "?")
        oos = cal.get("oos_valid")
        avg_d = cal.get("avg_decay_pct")
        n_pass = cal.get("n_folds_pass")
        rl_bl = cal.get("rl_blend")
        rl_reg = cal.get("rl_recommended", "?")
        det_reg = cal.get("regime", "?")
        run_ts = cal.get("run_timestamp", "")
        if run_ts:
            st.caption(f"Last calibration: **{run_ts[:19]}**")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Phase a", f"{best_a:.3f}" if isinstance(best_a, float) else str(best_a))
        c2.metric("Phase b", f"{best_b:.3f}" if isinstance(best_b, float) else str(best_b))
        c3.metric("OOS Valid", "YES" if oos else ("CAUTION" if oos is False else "N/A"))
        c4.metric("IS->OOS Decay",
                  f"{avg_d:.1f}%" if avg_d is not None else "N/A",
                  delta=">=10% warning" if (avg_d is not None and avg_d >= 10) else None,
                  delta_color="inverse")
        c5.metric("Folds Passing", f"{n_pass}/5" if n_pass is not None else "N/A")

        with st.expander("CAL-A -- Phase Grid Search Heatmap", expanded=True):
            st.markdown(f"Grid search a in [0.5, 2.0] x b in [0.8, 1.5] -- "
                        f"optimal: **a={best_a}**, **b={best_b}**")
            show_img("v4_phase_calibration_heatmap.png", "OOS Sharpe heatmap")

        with st.expander("CAL-B -- Walk-Forward OOS Validation"):
            show_img("v4_oos_validation.png", "IS vs OOS vs Dropout Sharpe per fold")

        with st.expander("CAL-C -- Regime Basis Weights & RL Q-Table"):
            if rl_bl is not None:
                st.markdown(f"Detected: **{det_reg}** | RL recommended: **{rl_reg}**  \n"
                            f"Blend: {(1-rl_bl)*100:.0f}% L2-opt + {rl_bl*100:.0f}% RL")
            fbw = cal.get("final_basis_weights", {})
            if fbw:
                bw_df = pd.DataFrame([fbw]).rename(index={0: "Final weight"})
                st.dataframe(bw_df.style.format("{:.4f}"), width="stretch")
            cw, cr = st.columns(2)
            with cw:
                show_img("v4_regime_weights_optimised.png", "L2-regularised weights per regime")
            with cr:
                show_img("v4_rl_qtable.png", "RL Q-table heatmap")

    st.divider()
    st.markdown("### |psi|^2 Wave Function Amplitudes")
    if cal:
        best_a = cal.get("best_a", "?")
        best_b = cal.get("best_b", "?")
    st.caption(f"theta(phi) = {best_a}*pi*(1-phi)^{best_b} -> |psi|^2 = cos^2(theta)")

    cwave, cz = st.columns(2)
    with cwave:
        show_img("v4_wave_amplitudes.png", "Top 40 holdings by |psi|^2")
    with cz:
        st.markdown("**Cointegration Z-score Signals**")
        show_img("v4_zscore_signals.png", "Spread Z-scores vs sector ETFs")

    st.divider()
    st.markdown("### IBKR Paper Trading Orders")
    orders = load_ibkr_orders()
    if orders is None:
        st.info("No ibkr_orders.json -- run Phase 8 Export.")
    else:
        df_orders = pd.DataFrame(orders)
        aum = cal.get("aum", 0) if cal else 0
        if df_orders.empty:
            st.info("Order book is empty.")
        else:
            st.caption(f"{len(df_orders)} orders | AUM: ${aum:,.0f}")
            st.dataframe(df_orders, width="stretch")
            st.download_button("Download ibkr_orders.json",
                               data=json.dumps(orders, indent=2),
                               file_name="ibkr_orders.json",
                               mime="application/json")

# -- TAB 7: QUANTUM 3D -----------------------------------------------------
with tab_q3d:
    st.markdown("### Quantum Wave-Function Visualizations\n"
                "> *Three interactive 3D plots exposing quantum interference mechanics.*")

    if not _HAS_PLOTLY:
        st.error("plotly not installed -- pip install plotly kaleido")
    else:
        try:
            from v4_quantum_3d_plots import generate_3d_visuals
            _q3d_ok = True
        except ImportError as ie:
            st.error(f"Could not import v4_quantum_3d_plots: {ie}")
            _q3d_ok = False

        if _q3d_ok:
            _pq, _pd = load_portfolio()
            _cq = load_cal_meta()
            _ba = float(_cq.get("best_a", 1.0)) if _cq else 1.0
            _bb = float(_cq.get("best_b", 1.0)) if _cq else 1.0
            _fw = (_cq.get("final_basis_weights",
                   {"momentum": 0.30, "safety": 0.30, "sortino": 0.25, "value": 0.15})
                   if _cq else
                   {"momentum": 0.30, "safety": 0.30, "sortino": 0.25, "value": 0.15})

            _snap = os.path.join(OUTPUT_DIR, "v4_3d_snapshot.json")
            if os.path.exists(_snap):
                try:
                    _mq = pd.read_json(_snap)
                    st.caption(f"Universe: **{len(_mq)} stocks** -- portfolio: {_pd}")
                except Exception:
                    _mq = _pq
            elif _pq is not None:
                _mq = _pq.copy()
                st.caption("Using portfolio holdings only (no universe snapshot).")
            else:
                _mq = None

            if _mq is None or _pq is None:
                st.warning("No portfolio data. Run the notebook first.")
            else:
                hc, bc = st.columns([5, 1])
                with hc:
                    st.markdown(f"Phase **a={_ba:.3f}**, **b={_bb:.3f}** -- "
                                f"{len(_pq)} holdings -- universe: {len(_mq)} stocks")
                if "_q3d_figs" not in st.session_state:
                    st.session_state["_q3d_figs"] = None
                with bc:
                    _render = st.button("Render 3D", help="Generate 3D plots (~2-5s)")

                if _render:
                    with st.spinner("Rendering 3D quantum visualizations..."):
                        try:
                            f1, f2, f3 = generate_3d_visuals(
                                _mq, _pq, best_a=_ba, best_b=_bb, basis_weights=_fw)
                            st.session_state["_q3d_figs"] = (f1, f2, f3)
                        except Exception as exc:
                            st.error(f"Render failed: {exc}")

                if st.session_state.get("_q3d_figs") is not None:
                    f1, f2, f3 = st.session_state["_q3d_figs"]

                    t1, t2, t3 = st.tabs([
                        "Interference Surface",
                        "Phase Trajectories",
                        "Energy Landscape",
                    ])
                    with t1:
                        st.markdown("#### Wave Interference Surface")
                        st.plotly_chart(f1, width="stretch")
                    with t2:
                        st.markdown("#### Phase Space Trajectories")
                        st.plotly_chart(f2, width="stretch")
                    with t3:
                        st.markdown("#### Energy Landscape")
                        st.plotly_chart(f3, width="stretch")

                    st.divider()
                    st.markdown("##### Export Interactive HTML")
                    ec1, ec2, ec3 = st.columns(3)
                    for col, fig, fname in [
                        (ec1, f1, "v4_3d_interference_surface.html"),
                        (ec2, f2, "v4_3d_phase_trajectories.html"),
                        (ec3, f3, "v4_3d_energy_landscape.html"),
                    ]:
                        with col:
                            st.download_button(
                                f"Download {fname}",
                                data=fig.to_html(include_plotlyjs="cdn", full_html=True).encode(),
                                file_name=fname, mime="text/html")
                else:
                    st.info("Click **Render 3D** to generate interactive 3D plots.")