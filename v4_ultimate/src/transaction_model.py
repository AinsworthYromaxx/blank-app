"""
Transaction Cost Model — V4
============================
Estimates realistic round-trip trading costs for a $10M AUM portfolio
rebalanced quarterly.

Cost model per trade:
    total_cost = commission + market_impact
    commission  = 0.1 bps flat per trade
    market_impact = max(0.10%, vol_60 / 10)   (Almgren-Chriss inspired)

Usage:
    from transaction_model import TransactionCostModel
    tcm = TransactionCostModel(aum=10_000_000)
    costs = tcm.estimate(old_weights, new_weights, price_data)
    report = tcm.report(costs)
"""

import numpy as np
import pandas as pd


class TransactionCostModel:
    """
    Estimates transaction costs for a portfolio rebalance.

    Parameters
    ----------
    aum              : float   — Assets under management in USD
    commission_bps   : float   — Flat commission per trade (basis points)
    impact_floor_pct : float   — Minimum market impact (%)
    impact_vol_div   : float   — vol_60 divisor for impact estimate
    """

    def __init__(
        self,
        aum: float = 10_000_000,
        commission_bps: float = 0.1,
        impact_floor_pct: float = 0.10,
        impact_vol_div: float = 10.0,
    ):
        self.aum = aum
        self.commission_bps = commission_bps
        self.impact_floor_pct = impact_floor_pct
        self.impact_vol_div = impact_vol_div

    # ── Core cost calculation ─────────────────────────────────────────────────
    def cost_per_trade(self, trade_value: float, vol_60: float) -> dict:
        """
        Estimate cost for a single trade.

        Parameters
        ----------
        trade_value : float  — Absolute USD value of the trade
        vol_60      : float  — 60-day annualised volatility (decimal, e.g. 0.25)

        Returns
        -------
        dict with keys: trade_value, commission, market_impact, total_cost, total_bps
        """
        commission    = trade_value * (self.commission_bps / 10_000)
        impact_pct    = max(self.impact_floor_pct / 100, vol_60 / self.impact_vol_div)
        market_impact = trade_value * impact_pct
        total_cost    = commission + market_impact
        total_bps     = (total_cost / trade_value * 10_000) if trade_value > 0 else 0

        return {
            'trade_value':    round(trade_value, 2),
            'commission':     round(commission, 2),
            'market_impact':  round(market_impact, 2),
            'total_cost':     round(total_cost, 2),
            'total_bps':      round(total_bps, 3),
        }

    # ── Portfolio-level estimate ──────────────────────────────────────────────
    def estimate(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        price_data: dict,
    ) -> pd.DataFrame:
        """
        Estimate rebalancing costs for a full portfolio transition.

        Parameters
        ----------
        old_weights : pd.Series  — Current weights indexed by ticker (0 for new entrants)
        new_weights : pd.Series  — Target weights indexed by ticker
        price_data  : dict       — {ticker: pd.Series of prices}

        Returns
        -------
        DataFrame with one row per traded ticker and cost breakdown columns,
        plus a 'TOTAL' summary row.
        """
        all_tickers = set(old_weights.index) | set(new_weights.index)
        rows = []

        for tk in sorted(all_tickers):
            old_w = float(old_weights.get(tk, 0.0))
            new_w = float(new_weights.get(tk, 0.0))
            delta_w = abs(new_w - old_w)

            if delta_w < 1e-6:
                continue

            trade_value = delta_w * self.aum

            # Get vol_60 from price data
            vol_60 = 0.20  # fallback 20%
            if tk in price_data:
                px = price_data[tk]
                if isinstance(px, pd.DataFrame):
                    px = px.iloc[:, 0]
                rets = px.dropna().pct_change().dropna()
                if len(rets) >= 60:
                    vol_60 = float(rets.iloc[-60:].std() * np.sqrt(252))

            action = 'BUY' if new_w > old_w else 'SELL'
            cost = self.cost_per_trade(trade_value, vol_60)

            rows.append({
                'ticker':        tk,
                'action':        action,
                'old_weight':    round(old_w * 100, 3),
                'new_weight':    round(new_w * 100, 3),
                'delta_weight':  round((new_w - old_w) * 100, 3),
                'trade_usd':     cost['trade_value'],
                'commission':    cost['commission'],
                'market_impact': cost['market_impact'],
                'total_cost':    cost['total_cost'],
                'cost_bps':      cost['total_bps'],
                'vol_60':        round(vol_60 * 100, 2),
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Summary row
        total_row = {
            'ticker':        'TOTAL',
            'action':        '',
            'old_weight':    df['old_weight'].sum(),
            'new_weight':    df['new_weight'].sum(),
            'delta_weight':  df['delta_weight'].abs().sum(),
            'trade_usd':     df['trade_usd'].sum(),
            'commission':    df['commission'].sum(),
            'market_impact': df['market_impact'].sum(),
            'total_cost':    df['total_cost'].sum(),
            'cost_bps':      (df['total_cost'].sum() / self.aum) * 10_000,
            'vol_60':        df['vol_60'].mean(),
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        return df

    # ── Annual drag estimate ──────────────────────────────────────────────────
    def annual_drag(self, rebalance_cost_pct: float, quarterly: bool = True) -> float:
        """
        Annualise the cost drag given a single rebalance cost.

        Parameters
        ----------
        rebalance_cost_pct : float — Cost of one rebalance as % of AUM
        quarterly          : bool  — If True, multiply by 4 (quarterly rebalance)

        Returns
        -------
        float — Estimated annual cost drag as % of AUM
        """
        freq = 4 if quarterly else 12
        return rebalance_cost_pct * freq

    # ── Summary report ────────────────────────────────────────────────────────
    def report(self, costs_df: pd.DataFrame) -> None:
        """Print a formatted cost summary."""
        if costs_df.empty:
            print("No trades — portfolio unchanged.")
            return

        total = costs_df[costs_df['ticker'] == 'TOTAL'].iloc[0]
        trades = costs_df[costs_df['ticker'] != 'TOTAL']

        print("=" * 55)
        print("  TRANSACTION COST REPORT")
        print("=" * 55)
        print(f"  AUM              : ${self.aum:>14,.0f}")
        print(f"  Trades           : {len(trades):>14}")
        print(f"  Turnover (1-way) : {total['delta_weight']:>13.1f}%")
        print(f"  Total trade USD  : ${total['trade_usd']:>13,.0f}")
        print(f"  Commission       : ${total['commission']:>13,.0f}")
        print(f"  Market impact    : ${total['market_impact']:>13,.0f}")
        print(f"  TOTAL COST       : ${total['total_cost']:>13,.0f}  ({total['cost_bps']:.2f} bps)")
        drag = self.annual_drag(total['total_cost'] / self.aum * 100)
        print(f"  Annualised drag  : {drag:>13.3f}% (if quarterly)")
        print("=" * 55)

        # Top 10 most expensive trades
        top = trades.nlargest(10, 'total_cost')[['ticker', 'action', 'delta_weight', 'total_cost', 'cost_bps']]
        if not top.empty:
            print("\n  Top 10 trades by cost:")
            print(top.to_string(index=False))

    # ── Capacity curve ────────────────────────────────────────────────────────

    def capacity_curve(
        self,
        portfolio_df: pd.DataFrame,
        aum_levels: list = None,
        output_path: str = '../output/v3/v4_capacity_curve.png',
    ) -> pd.DataFrame:
        """
        Estimate total-drag cost across a range of AUM levels.

        Impact model (square-root market-impact):
            impact_bps = max(10bps, sqrt(notional / ADV_20d) * vol_60 * 10_000 / 10)

        ADV is fetched once per ticker from ``yfinance.Ticker.info['averageVolume']``.
        Vol_60 is estimated from the last 60 daily returns.

        Parameters
        ----------
        portfolio_df : pd.DataFrame
            Must contain ``ticker`` and ``weight`` (%) columns.
        aum_levels : list of float, optional
            AUM values in USD to test.  Default: [1M, 5M, 10M, 50M, 100M].
        output_path : str
            Where to save the capacity curve chart.

        Returns
        -------
        pd.DataFrame
            Columns: aum_usd, total_impact_usd, drag_bps, implied_sharpe_cost
        """
        import math
        import matplotlib.pyplot as plt

        if aum_levels is None:
            aum_levels = [1e6, 5e6, 1e7, 5e7, 1e8]

        tickers = portfolio_df['ticker'].tolist()
        weights = portfolio_df.set_index('ticker')['weight']

        # Pre-fetch ADV and vol_60 once
        adv_map: dict  = {}
        vol_map: dict  = {}
        for tk in tickers:
            try:
                info = yf.Ticker(tk).info
                adv_map[tk] = float(info.get('averageVolume', 1e6) or 1e6)
                # Use regularMarketPrice as rough close to infer vol cheaply
                price = float(info.get('regularMarketPrice', 0) or 0)
                # Fallback vol: fetch 60-day history
                hist = yf.download(tk, period='3mo', progress=False)
                if isinstance(hist.columns, pd.MultiIndex):
                    hist = hist.droplevel(1, axis=1)
                closes = hist['Close'].dropna()
                if len(closes) >= 5:
                    vol_map[tk] = float(closes.pct_change().dropna().std())
                else:
                    vol_map[tk] = 0.02
            except Exception:
                adv_map[tk] = 1e6
                vol_map[tk] = 0.02

        records = []
        for aum in aum_levels:
            total_impact = 0.0
            for tk in tickers:
                w_pct   = float(weights.get(tk, 0))
                notional = aum * w_pct / 100.0
                adv      = adv_map.get(tk, 1e6)
                vol60    = vol_map.get(tk, 0.02)
                # Square-root market impact (Almgren-Chriss style)
                impact_frac = max(0.001, math.sqrt(notional / max(adv, 1)) * vol60 / 10.0)
                total_impact += notional * impact_frac
            drag_bps_val = (total_impact / aum) * 10_000
            # Implied Sharpe cost assuming 4 rebalances/yr
            implied_sharpe_cost = (drag_bps_val / 10_000) * 4 / 0.15
            records.append({
                'aum_usd':            aum,
                'total_impact_usd':   round(total_impact, 2),
                'drag_bps':           round(drag_bps_val, 2),
                'implied_sharpe_cost': round(implied_sharpe_cost, 3),
            })

        result_df = pd.DataFrame(records)

        # ── Chart ─────────────────────────────────────────────────────────────
        try:
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()

            x_labels = [f"${int(r['aum_usd'] / 1e6)}M" for _, r in result_df.iterrows()]
            x_pos = range(len(x_labels))

            ax1.bar(x_pos, result_df['drag_bps'], color='steelblue', alpha=0.7, label='Drag (bps)')
            ax2.plot(x_pos, result_df['implied_sharpe_cost'], color='tomato',
                     marker='o', linewidth=2, label='Implied Sharpe cost')

            # Annotate current AUM
            if self.aum in aum_levels:
                idx = aum_levels.index(self.aum)
                ax1.axvline(idx, color='orange', linestyle='--', linewidth=1.5,
                            label=f'Current AUM ${self.aum/1e6:.0f}M')

            ax1.set_xticks(list(x_pos))
            ax1.set_xticklabels(x_labels)
            ax1.set_xlabel('AUM Level')
            ax1.set_ylabel('Total Drag (bps / quarter)')
            ax2.set_ylabel('Implied Sharpe Cost (annualised)')
            ax1.set_title('Portfolio Capacity Curve  |  Square-Root Market Impact')

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

            import os as _os
            _os.makedirs(_os.path.dirname(_os.path.abspath(output_path)), exist_ok=True)
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            print(f"[capacity_curve] Chart saved: {output_path}")
        except Exception as exc:
            print(f"[capacity_curve] Chart skipped: {exc}")

        print(result_df.to_string(index=False))
        return result_df
