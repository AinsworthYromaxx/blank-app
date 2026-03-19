"""
Quantum Wave Portfolio Constructor
===================================
Replaces classical percentile ranking with quantum-inspired wave function
probability amplitudes to select and weight stocks.

Key concepts mapped from quantum mechanics:
- Wave function ψ(x): amplitude built from normalized stock metrics
- |ψ|² rule: probability of selecting a stock = |amplitude|²
- Constructive interference: multiple positive metrics amplify selection
- Destructive interference: conflicting signals cancel amplitude
- Superposition: a stock's state is a weighted sum of basis states
  (quality, momentum, value, safety)
- Measurement (collapse): final portfolio drawn proportional to |ψ|²

Usage (from StockPickerV3_Production.ipynb):
    import sys; sys.path.insert(0, '../src')
    from stock_analysis import ComprehensiveStockAnalyzer, check_regime
    from quantum_portfolio import QuantumPortfolioConstructor

    analyzer = ComprehensiveStockAnalyzer(cache_dir='../cache/v3')
    results = analyzer.run_full_analysis(min_market_cap=300e6, max_tickers=1000)

    qpc = QuantumPortfolioConstructor()
    portfolio = qpc.build(analyzer, n_holdings=50, max_position_pct=5)
    print(portfolio[['ticker', 'weight', 'psi_sq', 'sector']].head(20))
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# BASIS STATES
# Each represents a dimension of stock quality.
# Their weights define how strongly each dimension contributes to ψ.
# =============================================================================
BASIS_WEIGHTS = {
    'momentum':  0.30,   # trend / price momentum
    'safety':    0.30,   # low drawdown + high win rate
    'sortino':   0.25,   # risk-adjusted return quality
    'value':     0.15,   # inverse-volatility proxy for value
}


def _normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. Returns 0.5 for degenerate series."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-10:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _compute_metrics(price_data: dict) -> pd.DataFrame:
    """
    Compute per-stock metrics from a dict of price Series/DataFrames
    (same input format as ComprehensiveStockAnalyzer.price_data).
    Returns a DataFrame indexed by ticker.
    """
    rows = []
    for ticker, raw in tqdm(price_data.items(), desc="  Computing metrics"):
        try:
            if isinstance(raw, pd.DataFrame):
                p = raw['Close'] if 'Close' in raw.columns else raw.iloc[:, 0]
            else:
                p = raw
            p = p.dropna()
            if len(p) < 252:
                continue

            r = p.pct_change().dropna()

            # Annualised return & vol
            ann_ret  = r.mean() * 252
            ann_vol  = r.std() * np.sqrt(252)
            down_r   = r[r < 0]
            down_vol = down_r.std() * np.sqrt(252) if len(down_r) > 10 else ann_vol
            sortino  = (ann_ret - 0.04) / down_vol if down_vol > 1e-4 else 0.0

            # Win rate (monthly)
            monthly  = p.resample('ME').last().pct_change().dropna()
            win_rate = (monthly > 0).mean() if len(monthly) > 0 else 0.5

            # Max drawdown
            rolling_max = p.cummax()
            max_dd = ((p - rolling_max) / rolling_max).min()  # negative

            # Momentum (3 m + 6 m average)
            m3 = p.iloc[-1] / p.iloc[-63]  - 1 if len(p) >= 63  else 0.0
            m6 = p.iloc[-1] / p.iloc[-126] - 1 if len(p) >= 126 else 0.0
            momentum = (m3 + m6) / 2

            # 60-day realised vol (for inverse-vol weighting)
            vol_60 = r.iloc[-60:].std() * np.sqrt(252) if len(r) >= 60 else ann_vol

            rows.append({
                'ticker':   ticker,
                'sortino':  sortino,
                'win_rate': win_rate,
                'max_dd':   max_dd,       # [–1, 0]
                'momentum': momentum,
                'vol_60':   vol_60,
                'last_price': float(p.iloc[-1]),
            })
        except Exception:
            continue

    return pd.DataFrame(rows).set_index('ticker')


class QuantumPortfolioConstructor:
    """
    Quantum-inspired portfolio selector.

    Wave function construction
    --------------------------
    For each stock i we build a complex amplitude by decomposing its
    signal into four orthogonal basis states {|momentum⟩, |safety⟩,
    |sortino⟩, |value⟩}:

        ψᵢ = Σₖ  √wₖ · φₖᵢ · exp(i·θₖᵢ)

    where
      φₖᵢ ∈ [0,1]  – normalised score on basis k  (magnitude)
      θₖᵢ ∈ [0, π] – phase angle derived from signal consistency
                      (0 = perfectly aligned, π = anti-aligned)
      wₖ            – basis weight (see BASIS_WEIGHTS)

    The selection probability is the Born rule:
        Pᵢ = |ψᵢ|² / Σⱼ |ψⱼ|²

    Portfolio weights are then computed by modulating Pᵢ with an
    inverse-volatility factor and capping each position.
    """

    def __init__(self, basis_weights: dict = None):
        self.basis_weights = basis_weights or BASIS_WEIGHTS

    # ------------------------------------------------------------------
    def _build_wave_function(self, df: pd.DataFrame) -> pd.Series:
        """
        Return |ψᵢ|² for every stock in df.

        The real and imaginary parts of the amplitude accumulate
        contributions from each basis state.  A metric whose 3-month
        and 6-month sub-signals point in the same direction gives a
        phase angle near 0 → constructive interference.  Disagreement
        between sub-signals pushes the phase toward π → destructive.
        """

        n = len(df)
        psi_real = np.zeros(n)
        psi_imag = np.zeros(n)

        # ── momentum basis ──────────────────────────────────────────────
        # φ = normalised momentum score
        # θ = 0 when momentum is strongly positive, π when negative
        phi_mom   = _normalize(df['momentum']).values
        theta_mom = np.pi * (1.0 - phi_mom)   # high phi → small θ → constructive
        w_mom     = self.basis_weights['momentum']
        psi_real += np.sqrt(w_mom) * phi_mom * np.cos(theta_mom)
        psi_imag += np.sqrt(w_mom) * phi_mom * np.sin(theta_mom)

        # ── safety basis  (win rate + low drawdown) ─────────────────────
        # Invert drawdown: higher = better safety
        safety_score = (_normalize(df['win_rate']) + _normalize(-df['max_dd'])) / 2
        phi_safe   = safety_score.values
        theta_safe = np.pi * (1.0 - phi_safe)
        w_safe     = self.basis_weights['safety']
        psi_real += np.sqrt(w_safe) * phi_safe * np.cos(theta_safe)
        psi_imag += np.sqrt(w_safe) * phi_safe * np.sin(theta_safe)

        # ── sortino basis ────────────────────────────────────────────────
        phi_sort   = _normalize(df['sortino']).values
        theta_sort = np.pi * (1.0 - phi_sort)
        w_sort     = self.basis_weights['sortino']
        psi_real += np.sqrt(w_sort) * phi_sort * np.cos(theta_sort)
        psi_imag += np.sqrt(w_sort) * phi_sort * np.sin(theta_sort)

        # ── value basis  (inverse volatility as cheapness proxy) ─────────
        phi_val   = _normalize(1.0 / df['vol_60'].replace(0, np.nan).fillna(df['vol_60'].median())).values
        theta_val = np.pi * (1.0 - phi_val)
        w_val     = self.basis_weights['value']
        psi_real += np.sqrt(w_val) * phi_val * np.cos(theta_val)
        psi_imag += np.sqrt(w_val) * phi_val * np.sin(theta_val)

        psi_sq = psi_real**2 + psi_imag**2
        return pd.Series(psi_sq, index=df.index, name='psi_sq')

    # ------------------------------------------------------------------
    def _collapse(
        self,
        metrics: pd.DataFrame,
        psi_sq: pd.Series,
        n_holdings: int,
        max_position_pct: float,
        sector_map: dict,
    ) -> pd.DataFrame:
        """
        'Measurement' step: collapse the quantum superposition into a
        definite portfolio by:
          1. Ranking stocks by |ψ|² and taking the top n_holdings
          2. Weighting by Pᵢ · (1 / σᵢ)  (probability × inverse-vol)
          3. Capping each weight and renormalising
        """
        df = metrics.copy()
        df['psi_sq'] = psi_sq

        # Filter obviously bad stocks (negative sortino, huge drawdown)
        df = df[(df['sortino'] > 0) & (df['max_dd'] > -0.60)]

        # Select top n_holdings by |ψ|²
        top = df.nlargest(n_holdings, 'psi_sq').copy()

        # Born-rule probabilities within selection
        top['prob'] = top['psi_sq'] / top['psi_sq'].sum()

        # Blend with inverse-vol for position sizing
        inv_vol = 1.0 / top['vol_60'].replace(0, np.nan).fillna(top['vol_60'].median())
        inv_vol_norm = inv_vol / inv_vol.sum()
        raw_weight = 0.5 * top['prob'] + 0.5 * inv_vol_norm

        # Cap and renormalise
        cap = max_position_pct / 100.0
        capped = raw_weight.clip(upper=cap)
        top['weight'] = capped / capped.sum()

        top['sector'] = top.index.map(sector_map).fillna('Unknown')

        # --- Iterative sector cap (35% max per sector) -------------------
        # Simple proportional redistribution can fail with 2-sector concentration.
        # Iteratively trim the heaviest overweight sector until all sectors comply.
        SECTOR_CAP = 0.35
        MAX_ITER = 5
        for iteration in range(MAX_ITER):
            sector_sums = top.groupby('sector')['weight'].sum()
            overweight = sector_sums[sector_sums > SECTOR_CAP]
            if overweight.empty:
                break
            # Trim the biggest overweight sector's largest position by 10%
            worst_sector = overweight.idxmax()
            mask = top['sector'] == worst_sector
            largest_idx = top.loc[mask, 'weight'].idxmax()
            top.loc[largest_idx, 'weight'] *= 0.90
            top['weight'] = top['weight'] / top['weight'].sum()  # renormalise
        else:
            # Ran out of iterations
            still_over = top.groupby('sector')['weight'].sum()
            still_over = still_over[still_over > SECTOR_CAP]
            if not still_over.empty:
                print(f"WARNING: sector concentration unresolvable after {MAX_ITER} iterations: "
                      f"{still_over.to_dict()}")
        # ------------------------------------------------------------------

        top = top.reset_index().rename(columns={'index': 'ticker'})
        return top.sort_values('weight', ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    def build(
        self,
        analyzer,
        n_holdings: int = 50,
        max_position_pct: float = 5.0,
    ) -> pd.DataFrame:
        """
        Build a quantum-wave portfolio from an already-run analyzer.

        Parameters
        ----------
        analyzer         : ComprehensiveStockAnalyzer (post run_full_analysis)
        n_holdings       : target number of stocks
        max_position_pct : hard cap per position (%)

        Returns
        -------
        DataFrame with columns:
            ticker, weight, psi_sq, prob, sortino, win_rate,
            max_dd, momentum, vol_60, last_price, sector
        """
        if not analyzer.price_data:
            raise ValueError("Run analyzer.run_full_analysis() first.")

        print("Quantum portfolio construction")
        print("=" * 50)

        # 1. Metrics
        print("[1/3] Computing stock metrics...")
        metrics = _compute_metrics(analyzer.price_data)
        print(f"      {len(metrics)} stocks with sufficient history")

        # 2. Wave function
        print("[2/3] Building wave function amplitudes (ψ)...")
        psi_sq = self._build_wave_function(metrics)
        total_prob = psi_sq.sum()
        top5 = psi_sq.nlargest(5)
        print(f"      Total probability mass: {total_prob:.4f}")
        print(f"      Top 5 by |ψ|²: {', '.join(f'{t}({v:.4f})' for t, v in top5.items())}")

        # 3. Collapse / measure
        print("[3/3] Collapsing superposition → portfolio...")
        sector_map = {}
        if analyzer.metrics is not None and 'ticker' in analyzer.metrics.columns:
            sector_map = dict(zip(analyzer.metrics['ticker'], analyzer.metrics.get('sector', pd.Series(dtype=str))))

        portfolio = self._collapse(metrics, psi_sq, n_holdings, max_position_pct, sector_map)

        print(f"\n✓ Portfolio: {len(portfolio)} stocks")
        print(f"  Weight range: {portfolio['weight'].min()*100:.2f}% – {portfolio['weight'].max()*100:.2f}%")
        print(f"  Sectors: {portfolio['sector'].nunique()}")
        return portfolio

    # ------------------------------------------------------------------
    def sector_breakdown(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        return (
            portfolio.groupby('sector')['weight']
            .sum()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
            .rename('weight_%')
            .reset_index()
        )

    # ------------------------------------------------------------------
    def export(self, portfolio: pd.DataFrame, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        portfolio.to_csv(path, index=False)
        print(f"Exported to {path}")


# =============================================================================
# V4: BOOTSTRAP MONTE CARLO  (replaces Gaussian MC)
# =============================================================================

def bootstrap_mc(
    portfolio: pd.DataFrame,
    price_data: dict,
    n_sims: int = 1000,
    horizon: int = 252,
    block_size: int = 21,
    initial: float = 100_000,
    psi_modulate: bool = True,
    rf_daily: float = 0.04 / 252,
    seed: int = 42,
) -> dict:
    """
    Block-bootstrap Monte Carlo for a given portfolio.

    Unlike Gaussian MC, this resamples actual 21-day return blocks from
    history, preserving autocorrelation and fat-tail structure.

    |ψ|²-modulation (when psi_modulate=True):
        The number of bootstrap iterations for each stock is inversely
        proportional to psi_sq — high-confidence stocks see less
        random reshuffling of their block order.  Concretely, for each
        stock we pre-select a subset of block start indices with
        probability proportional to psi_sq; low-ψ stocks draw from a
        larger, more random pool.

    Parameters
    ----------
    portfolio      : DataFrame with columns ticker, weight, psi_sq
    price_data     : dict {ticker: pd.Series}
    n_sims         : number of Monte Carlo paths
    horizon        : trading days per path
    block_size     : length of each return block (21 ≈ 1 month)
    initial        : starting portfolio value ($)
    psi_modulate   : whether to apply |ψ|² noise scaling
    rf_daily       : daily risk-free rate for Sharpe calculation
    seed           : RNG seed

    Returns
    -------
    dict with keys:
        paths      : np.ndarray (n_sims, horizon+1)
        returns    : 1-year final returns (%)
        mdd        : max drawdowns per path (%)
        var95      : VaR 95%
        cvar95     : CVaR 95%
        sharpe_med : Sharpe of median path
        stats      : summary dict
    """
    rng = np.random.default_rng(seed)

    # Build per-stock daily-return arrays
    tickers = portfolio['ticker'].tolist()
    weights = portfolio.set_index('ticker')['weight']

    ret_arrays: dict = {}
    for tk in tickers:
        if tk not in price_data:
            continue
        px = price_data[tk]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        px = px.dropna()
        if len(px) < block_size * 5:
            continue
        ret_arrays[tk] = px.pct_change().dropna().values

    valid = [tk for tk in tickers if tk in ret_arrays]
    if not valid:
        raise ValueError("No valid price data for portfolio tickers.")

    wts = weights[valid]
    wts = wts / wts.sum()

    # psi_sq for modulation
    psi = portfolio.set_index('ticker')['psi_sq']
    psi_norm = (psi - psi.min()) / (psi.max() - psi.min() + 1e-12)

    paths = np.ones((n_sims, horizon + 1)) * initial

    for sim in range(n_sims):
        day = 0
        port_daily = np.zeros(horizon)

        for tk in valid:
            hist = ret_arrays[tk]
            n_blocks_needed = int(np.ceil(horizon / block_size))

            # Total possible block start indices
            max_start = len(hist) - block_size
            if max_start <= 0:
                continue
            starts = np.arange(max_start)

            if psi_modulate:
                # High psi → prefer blocks from the recent tail (more stable)
                # Low psi  → draw uniformly (more random)
                p_val = float(psi_norm.get(tk, 0.5))
                recent_cutoff = max(1, int(max_start * (0.5 + 0.4 * p_val)))
                pool = np.concatenate([
                    starts[-recent_cutoff:],                  # recent blocks
                    rng.choice(starts, size=len(starts) - recent_cutoff, replace=False)
                ]) if recent_cutoff < max_start else starts
            else:
                pool = starts

            chosen_starts = rng.choice(pool, size=n_blocks_needed, replace=True)
            stock_seq = np.concatenate([hist[s:s + block_size] for s in chosen_starts])[:horizon]
            port_daily += wts[tk] * stock_seq

        paths[sim, 1:] = initial * np.cumprod(1 + port_daily)

    # Statistics
    final_rets = (paths[:, -1] / initial - 1) * 100

    cummax = np.maximum.accumulate(paths, axis=1)
    mdd = ((paths - cummax) / cummax * 100).min(axis=1)

    var95  = float(np.percentile(final_rets, 5))
    cvar95 = float(final_rets[final_rets <= var95].mean())

    median_path = np.median(paths, axis=0)
    med_daily   = np.diff(median_path) / median_path[:-1]
    excess      = med_daily - rf_daily
    sharpe_med  = float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(252))

    stats = {
        'median_cagr_pct':  round(float(np.median(final_rets)), 2),
        'std_pct':          round(float(final_rets.std()), 2),
        'sharpe_median':    round(sharpe_med, 3),
        'var95_pct':        round(var95, 2),
        'cvar95_pct':       round(cvar95, 2),
        'mdd_p50_pct':      round(float(np.median(mdd)), 2),
        'mdd_p95_pct':      round(float(np.percentile(mdd, 5)), 2),
        'win_rate_pct':     round(float((final_rets > 0).mean() * 100), 1),
    }

    return {
        'paths':     paths,
        'returns':   final_rets,
        'mdd':       mdd,
        'var95':     var95,
        'cvar95':    cvar95,
        'sharpe_med': sharpe_med,
        'stats':     stats,
    }


# =============================================================================
# V4: RISK-PARITY WEIGHTING
# =============================================================================

def risk_parity_weights(
    tickers: list,
    price_data: dict,
    lookback: int = 60,
    max_position_pct: float = 5.0,
) -> pd.Series:
    """
    Compute risk-parity weights for a list of tickers.

    Each stock is sized so that its marginal contribution to portfolio
    risk is equal across all holdings (equal risk contribution, ERC).

    Uses scipy.optimize to solve:
        min  Σᵢ (w_i * (Σw)ᵢ - target_risk)²
    subject to: Σ w_i = 1, w_i ≥ 0

    where (Σw)ᵢ is the i-th element of the risk contribution vector.

    Parameters
    ----------
    tickers         : list of ticker strings
    price_data      : dict {ticker: pd.Series}
    lookback        : days of history to estimate covariance
    max_position_pct: hard cap per position (%)

    Returns
    -------
    pd.Series indexed by ticker with weights summing to 1
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback to inverse-vol if scipy not available
        vols = {}
        for tk in tickers:
            if tk in price_data:
                px = price_data[tk]
                if isinstance(px, pd.DataFrame):
                    px = px.iloc[:, 0]
                rets = px.dropna().pct_change().dropna()
                if len(rets) >= lookback:
                    vols[tk] = float(rets.iloc[-lookback:].std())
        if not vols:
            return pd.Series(1 / len(tickers), index=tickers)
        inv_v = pd.Series({tk: 1 / v for tk, v in vols.items()})
        inv_v = inv_v / inv_v.sum()
        return inv_v.clip(upper=max_position_pct / 100).pipe(lambda s: s / s.sum())

    # Build return matrix
    ret_df = pd.DataFrame()
    for tk in tickers:
        if tk in price_data:
            px = price_data[tk]
            if isinstance(px, pd.DataFrame):
                px = px.iloc[:, 0]
            rets = px.dropna().pct_change().dropna()
            if len(rets) >= lookback:
                ret_df[tk] = rets.iloc[-lookback:]

    valid_tickers = ret_df.columns.tolist()
    if len(valid_tickers) < 2:
        return pd.Series(1 / len(tickers), index=tickers)

    cov = ret_df.cov().values
    n   = len(valid_tickers)
    cap = max_position_pct / 100

    def risk_contributions(w):
        port_var  = w @ cov @ w
        marginal  = cov @ w
        return w * marginal / (port_var + 1e-12)

    def objective(w):
        rc    = risk_contributions(w)
        target = 1.0 / n
        return float(np.sum((rc - target) ** 2))

    w0 = np.ones(n) / n
    bounds = [(0.0, cap)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    result = minimize(
        objective, w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 500},
    )

    w_opt = result.x if result.success else w0
    w_opt = np.clip(w_opt, 0, cap)
    w_opt /= w_opt.sum()

    return pd.Series(w_opt, index=valid_tickers)


# =============================================================================
# V4: QUANTUM PORTFOLIO CONSTRUCTOR WITH DYNAMIC WEIGHTS + RISK-PARITY BLEND
# =============================================================================

class QuantumPortfolioConstructorV4(QuantumPortfolioConstructor):
    """
    V4 extension of QuantumPortfolioConstructor that adds:
    - Dynamic basis weights from RegimeDetector
    - Risk-parity blending (40% Born + 30% inv-vol + 30% RP)
    - Cointegration Z-score column on output

    Usage:
        from regime_detector import RegimeDetector
        from quantum_portfolio import QuantumPortfolioConstructorV4

        rd  = RegimeDetector()
        qpc = QuantumPortfolioConstructorV4(regime_detector=rd)
        portfolio = qpc.build(analyzer, n_holdings=50)
    """

    WEIGHT_BLEND = {'born': 0.40, 'inv_vol': 0.30, 'risk_parity': 0.30}

    def __init__(self, regime_detector=None, basis_weights: dict = None):
        self.regime_detector = regime_detector
        if basis_weights:
            super().__init__(basis_weights=basis_weights)
        elif regime_detector is not None:
            dynamic_w = regime_detector.get_basis_weights()
            super().__init__(basis_weights=dynamic_w)
            print(f"  Regime: {regime_detector.regime()}  →  basis weights: {dynamic_w}")
        else:
            super().__init__()

    def build(
        self,
        analyzer,
        n_holdings: int = 50,
        max_position_pct: float = 5.0,
        add_zscore: bool = True,
    ) -> pd.DataFrame:
        """
        Build a V4 quantum portfolio with risk-parity blending.

        Parameters
        ----------
        analyzer         : ComprehensiveStockAnalyzer (run_full_analysis done)
        n_holdings       : target portfolio size
        max_position_pct : hard cap per position (%)
        add_zscore       : compute cointegration Z-scores for output df

        Returns
        -------
        DataFrame with all V3 columns + psi_percentile, regime, zscore
        """
        if not analyzer.price_data:
            raise ValueError("Run analyzer.run_full_analysis() first.")

        print("Quantum Portfolio V4 — construction")
        print("=" * 50)

        # ── 1. Metrics + wave function ─────────────────────────────────
        print("[1/4] Computing metrics & wave function...")
        metrics = _compute_metrics(analyzer.price_data)
        psi_sq  = self._build_wave_function(metrics)
        print(f"      Universe: {len(metrics)} stocks")

        # ── 2. Top-N selection by |ψ|² ─────────────────────────────────
        df = metrics.copy()
        df['psi_sq'] = psi_sq
        df = df[(df['sortino'] > 0) & (df['max_dd'] > -0.60)]
        top = df.nlargest(n_holdings, 'psi_sq').copy()

        sector_map = {}
        if analyzer.metrics is not None and 'ticker' in analyzer.metrics.columns:
            sector_map = dict(zip(analyzer.metrics['ticker'],
                                  analyzer.metrics.get('sector', pd.Series(dtype=str))))
        top['sector'] = top.index.map(sector_map).fillna('Unknown')

        # ── 3. Three-way weight blending ───────────────────────────────
        print("[2/4] Three-way weighting (Born + Inv-Vol + Risk-Parity)...")
        tickers_sel = top.index.tolist()

        # Born-rule probability
        born_w     = top['psi_sq'] / top['psi_sq'].sum()

        # Inverse-vol
        inv_v      = 1.0 / top['vol_60'].replace(0, np.nan).fillna(top['vol_60'].median())
        inv_vol_w  = inv_v / inv_v.sum()

        # Risk parity
        rp_w_series = risk_parity_weights(
            tickers_sel, analyzer.price_data, max_position_pct=max_position_pct
        )
        rp_w = rp_w_series.reindex(tickers_sel).fillna(inv_vol_w)
        rp_w = rp_w / rp_w.sum()

        blend = self.WEIGHT_BLEND
        raw_w = (
            blend['born']       * born_w
            + blend['inv_vol']  * inv_vol_w
            + blend['risk_parity'] * rp_w
        )

        cap = max_position_pct / 100
        capped = raw_w.clip(upper=cap)
        top['weight'] = (capped / capped.sum()).values

        # Born probability column
        top['prob'] = born_w.values
        top = top.reset_index().rename(columns={'index': 'ticker'})

        # ── 4. Cointegration Z-scores ──────────────────────────────────
        top['zscore'] = 0.0
        top['regime'] = self.regime_detector.regime() if self.regime_detector else 'NEUTRAL'

        if add_zscore:
            print("[3/4] Computing cointegration Z-scores (may take ~2 min)...")
            try:
                from stock_analysis import QuarterlyRebalancer
                qr = QuarterlyRebalancer(top, lookback_days=60, zscore_threshold=2.0)
                sig = qr.generate_signals(verbose=False)
                zscore_map = dict(zip(sig['ticker'], sig['zscore']))
                top['zscore'] = top['ticker'].map(zscore_map).fillna(0.0)
            except Exception as e:
                print(f"      Z-score skipped: {e}")
        else:
            print("[3/4] Z-scores skipped (add_zscore=False)")

        # psi_percentile within full universe
        top['psi_percentile'] = (psi_sq.rank(pct=True) * 100).reindex(top['ticker']).values

        print("[4/4] Done.")
        top = top.sort_values('weight', ascending=False).reset_index(drop=True)

        print(f"\n✓ V4 Portfolio: {len(top)} stocks")
        print(f"  Regime       : {top['regime'].iloc[0]}")
        print(f"  Weight range : {top['weight'].min()*100:.2f}% – {top['weight'].max()*100:.2f}%")
        print(f"  Sectors      : {top['sector'].nunique()}")
        print(f"  Avg ψ-pctile : {top['psi_percentile'].mean():.1f}th")
        return top
