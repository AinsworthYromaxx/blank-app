"""
V4 Interactive Compute Engine
=================================
Cached functions for real-time regime / AUM / Monte Carlo recomputation.

All heavy functions accept JSON strings (not DataFrames) as arguments so that
Streamlit's @st.cache_data can hash them correctly.

Functions
---------
recompute_psi(portfolio_json, regime)         -> DataFrame
fast_mc(portfolio_json, n_sims, horizon, aum) -> dict
compute_ibkr_orders(portfolio, aum)           -> DataFrame
regime_movers(portfolio, new_regime)          -> DataFrame
format_aum(aum)                               -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ── Regime configurations ─────────────────────────────────────────────────────

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "BULL":    {"momentum": 0.40, "safety": 0.25, "sortino": 0.20, "value": 0.15},
    "NEUTRAL": {"momentum": 0.30, "safety": 0.30, "sortino": 0.25, "value": 0.15},
    "BEAR":    {"momentum": 0.10, "safety": 0.45, "sortino": 0.20, "value": 0.25},
}

REGIME_META: dict[str, dict] = {
    "BULL":    {"icon": "🟢", "color": "#00d26a", "tagline": "Chase winners · Momentum dominates"},
    "NEUTRAL": {"icon": "🟡", "color": "#ffd700", "tagline": "Balanced · Equal-weight factors"},
    "BEAR":    {"icon": "🔴", "color": "#ff4b4b", "tagline": "Capital preservation · Safety first"},
}

SECTOR_COLORS: dict[str, str] = {
    "Technology":             "#4a9eff",
    "Healthcare":             "#72d572",
    "Financial Services":     "#ffb347",
    "Consumer Cyclical":      "#ff6b9d",
    "Consumer Discretionary": "#ff6b9d",
    "Consumer Defensive":     "#c3a6ff",
    "Consumer Staples":       "#c3a6ff",
    "Industrials":            "#40e0d0",
    "Real Estate":            "#f9ca24",
    "Communication Services": "#f0e68c",
    "Utilities":              "#7ec8e3",
    "Energy":                 "#ff9ff3",
    "Basic Materials":        "#a0c4ff",
    "Materials":              "#a0c4ff",
}

_DEFAULT_COLOR = "#aaaaaa"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rank_norm(s: pd.Series) -> pd.Series:
    """Percentile rank → [0, 1]."""
    return s.rank(pct=True, na_option="bottom").clip(0.0, 1.0)


def format_aum(aum: float) -> str:
    """Format AUM as human-readable string (e.g. €10.0M)."""
    if aum >= 1e9:
        return f"€{aum/1e9:.2f}B"
    return f"€{aum/1e6:.1f}M"


# ── Quantum ψ recomputation ───────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def recompute_psi(portfolio_json: str, regime: str) -> pd.DataFrame:
    """
    Re-build |ψ|² using regime-specific basis weights on the portfolio's
    own metric columns (sortino / win_rate / max_dd / momentum / vol_60).

    Returns DataFrame with [ticker, new_weight, psi_sq, delta_weight].
    """
    df = pd.read_json(portfolio_json)
    if "ticker" in df.columns:
        df = df.set_index("ticker")

    bw = REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["NEUTRAL"])
    n  = len(df)

    psi_r = np.zeros(n)
    psi_i = np.zeros(n)

    def _add_basis(phi: np.ndarray, w: float) -> None:
        theta = np.pi * (1.0 - phi)
        psi_r[:] += np.sqrt(w) * phi * np.cos(theta)
        psi_i[:] += np.sqrt(w) * phi * np.sin(theta)

    # momentum
    phi = _rank_norm(df["momentum"]).values if "momentum" in df.columns else np.full(n, 0.5)
    _add_basis(phi, bw["momentum"])

    # safety = (win_rate + |max_dd|) / 2
    if "win_rate" in df.columns and "max_dd" in df.columns:
        phi = ((_rank_norm(df["win_rate"]) + _rank_norm(-df["max_dd"])) / 2).values
    elif "win_rate" in df.columns:
        phi = _rank_norm(df["win_rate"]).values
    else:
        phi = np.full(n, 0.5)
    _add_basis(phi, bw["safety"])

    # sortino
    phi = _rank_norm(df["sortino"]).values if "sortino" in df.columns else np.full(n, 0.5)
    _add_basis(phi, bw["sortino"])

    # value (inverse-volatility)
    if "vol_60" in df.columns:
        iv  = 1.0 / df["vol_60"].replace(0, np.nan).fillna(df["vol_60"].median())
        phi = _rank_norm(iv).values
    else:
        phi = np.full(n, 0.5)
    _add_basis(phi, bw["value"])

    psi_sq = pd.Series(psi_r**2 + psi_i**2, index=df.index)

    # inverse-vol blending for sizing
    if "vol_60" in df.columns:
        iv_norm = iv / iv.sum()
    else:
        iv_norm = pd.Series(1.0 / n, index=df.index)

    prob  = psi_sq / psi_sq.sum()
    raw   = 0.5 * prob + 0.5 * iv_norm
    capped = raw.clip(upper=0.05)
    new_w  = capped / capped.sum()

    base_w = df["weight"] if "weight" in df.columns \
             else pd.Series(1.0 / n, index=df.index)

    return pd.DataFrame({
        "ticker":       psi_sq.index,
        "new_weight":   new_w.values,
        "psi_sq":       psi_sq.values,
        "delta_weight": (new_w - base_w.reindex(new_w.index).fillna(0)).values,
    }).reset_index(drop=True)


# ── Fast vectorised Monte Carlo ───────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fast_mc(
    portfolio_json: str,
    n_sims: int,
    horizon_days: int,
    aum: float,
    seed: int = 42,
) -> dict:
    """
    Vectorised parametric GBM Monte Carlo.

    μ and σ are estimated from the portfolio's weighted Sortino ratio
    and vol_60.  A 0.80% annual transaction-cost drag is applied.
    No network calls — completes in < 500 ms for 5 000 sims × 1 260 days.

    Returns
    -------
    dict with keys:
        paths     : {p5, p25, p50, p75, p95}  each np.ndarray (horizon_days+1,)
        x_years   : np.ndarray — time axis in years
        final_rets: np.ndarray — final % returns across all sims
        mdd       : np.ndarray — max-drawdown per sim (%)
        stats     : summary dict
    """
    p   = pd.read_json(portfolio_json)
    rng = np.random.default_rng(seed)

    wts = p["weight"] / p["weight"].sum() if "weight" in p.columns \
          else pd.Series(1.0 / len(p), index=p.index)

    # ── Portfolio-level μ, σ ──────────────────────────────────────────────────
    if "sortino" in p.columns and "vol_60" in p.columns:
        vol_w      = float((p["vol_60"] * wts).sum())
        sortino_w  = float((p["sortino"].clip(-5, 10) * wts).sum())
        downside_v = vol_w * 0.65
        ann_mu     = sortino_w * downside_v + 0.04
        port_vol   = vol_w * 0.55          # diversification factor
    else:
        ann_mu   = 0.17
        port_vol = 0.14

    tc_annual = 0.008                      # 0.80 % cost drag
    ann_mu   -= tc_annual
    daily_mu  = ann_mu / 252
    daily_sig = port_vol / np.sqrt(252)

    # ── Vectorised GBM ────────────────────────────────────────────────────────
    Z      = rng.standard_normal((n_sims, horizon_days))
    log_r  = (daily_mu - 0.5 * daily_sig**2) + daily_sig * Z
    cum_r  = np.cumsum(log_r, axis=1)
    paths  = aum * np.concatenate(
        [np.ones((n_sims, 1)), np.exp(cum_r)], axis=1
    )

    bands = {f"p{pct}": np.percentile(paths, pct, axis=0)
             for pct in [5, 25, 50, 75, 95]}

    # ── Statistics ────────────────────────────────────────────────────────────
    final_rets = (paths[:, -1] / aum - 1) * 100
    cum_max    = np.maximum.accumulate(paths, axis=1)
    mdd_paths  = (paths - cum_max) / cum_max * 100
    mdd        = mdd_paths.min(axis=1)

    var95  = float(np.percentile(final_rets, 5))
    cvar95 = float(final_rets[final_rets <= var95].mean())

    med_path  = bands["p50"]
    med_daily = np.diff(med_path) / med_path[:-1]
    rf_d      = 0.04 / 252
    sharpe    = float((med_daily - rf_d).mean() / (med_daily.std() + 1e-12) * np.sqrt(252))

    years    = horizon_days / 252
    ann_cagr = (float(np.median(paths[:, -1])) / aum) ** (1.0 / max(years, 0.01)) - 1

    return {
        "paths":      bands,
        "x_years":    np.linspace(0, years, horizon_days + 1),
        "final_rets": final_rets,
        "mdd":        mdd,
        "stats": {
            "sharpe":      round(sharpe, 3),
            "var95":       round(var95, 2),
            "cvar95":      round(cvar95, 2),
            "ann_cagr":    round(ann_cagr * 100, 2),
            "mdd_median":  round(float(np.median(mdd)), 2),
            "mdd_p95":     round(float(np.percentile(mdd, 5)), 2),
            "win_rate":    round(float((final_rets > 0).mean() * 100), 1),
            "tc_drag_pct": round(tc_annual * 100, 2),
        },
    }


# ── IBKR order book ───────────────────────────────────────────────────────────

def compute_ibkr_orders(portfolio: pd.DataFrame, aum: float) -> pd.DataFrame:
    """
    Generate IBKR-style order book with AUM-scaled share counts and cost
    estimates using an Almgren-Chriss inspired market-impact model.
    """
    df = portfolio.copy()
    df["usd_value"] = (df["weight"] * aum).round(2)

    if "last_price" in df.columns:
        px = df["last_price"].replace(0, np.nan).fillna(100.0)
        df["shares"] = (df["usd_value"] / px).round(0).astype(int)
        df["price"]  = px.round(2)
    else:
        df["shares"] = 0
        df["price"]  = 0.0

    # Cost: 0.1 bps commission + vol_60 / 10 market impact
    if "vol_60" in df.columns:
        df["cost_bps"] = (1.0 + df["vol_60"] * 100.0 / 10.0).round(3)
    else:
        df["cost_bps"] = 1.5

    df["est_cost_usd"] = (df["usd_value"] * df["cost_bps"] / 10_000).round(2)
    df["side"]         = "BUY"
    df["order_type"]   = "MKT"
    df["currency"]     = "USD"

    keep = [c for c in [
        "ticker", "sector", "side", "shares", "price",
        "order_type", "usd_value", "weight", "cost_bps",
        "est_cost_usd", "currency",
    ] if c in df.columns]
    return df[keep].sort_values("usd_value", ascending=False).reset_index(drop=True)


# ── Regime impact movers ──────────────────────────────────────────────────────

def regime_movers(portfolio: pd.DataFrame, new_regime: str) -> pd.DataFrame:
    """
    Return top-5 gainers and top-5 losers when switching to new_regime.
    Each row has [ticker, sector, base_wt, new_wt, delta_pct].
    """
    try:
        new_df   = recompute_psi(portfolio.to_json(), new_regime)
        new_df   = new_df.set_index("ticker")
        base_wts = portfolio.set_index("ticker")["weight"]
        new_wts  = new_df["new_weight"].reindex(base_wts.index).fillna(0)

        delta = (new_wts - base_wts).dropna()
        out   = pd.DataFrame({
            "ticker":     delta.index,
            "base_wt":    (base_wts.reindex(delta.index) * 100).round(2),
            "new_wt":     (new_wts.reindex(delta.index) * 100).round(2),
            "delta_pct":  (delta * 100).round(3),
        })
        if "sector" in portfolio.columns:
            out["sector"] = portfolio.set_index("ticker")["sector"].reindex(out["ticker"].values).values

        gainers = out.nlargest(5, "delta_pct")
        losers  = out.nsmallest(5, "delta_pct")
        combined = pd.concat([gainers, losers]).drop_duplicates("ticker")
        return combined.sort_values("delta_pct", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
