"""
Regime Detector — V4
====================
Classifies the current macro regime (BULL / NEUTRAL / BEAR) using:
  - VIX level
  - 10Y-2Y yield curve (FRED: T10Y2Y)
  - High-yield credit spread (FRED: BAMLH0A0HYM2)
  - SPY vs 200-day SMA

The regime drives dynamic basis-state weights for the quantum wave function.

Usage:
    from regime_detector import RegimeDetector
    rd = RegimeDetector()
    regime = rd.detect()         # 'BULL' | 'NEUTRAL' | 'BEAR'
    weights = rd.get_basis_weights()
"""

import os
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# Dynamic basis weights per regime
# These govern how strongly each quantum basis state contributes to |ψ|²
REGIME_WEIGHTS = {
    'BULL': {
        'momentum': 0.40,  # chase winners in bull markets
        'safety':   0.25,
        'sortino':  0.20,
        'value':    0.15,
    },
    'NEUTRAL': {
        'momentum': 0.30,  # balanced default
        'safety':   0.30,
        'sortino':  0.25,
        'value':    0.15,
    },
    'BEAR': {
        'momentum': 0.10,  # avoid momentum traps
        'safety':   0.45,  # prioritise capital preservation
        'sortino':  0.20,
        'value':    0.25,  # value tends to hold better
    },
}


class RegimeDetector:
    """
    Classifies the current macro regime using four indicators:

    Indicator         | Bear signal           | Source
    ------------------|-----------------------|--------
    VIX               | > 25                  | yfinance ^VIX
    Yield curve       | 10Y-2Y < 0            | FRED T10Y2Y
    Credit spread     | HY spread > 5%        | FRED BAMLH0A0HYM2
    SPY/SMA           | SPY < 200-day SMA     | yfinance SPY

    Score = weighted sum of bear signals (-1 each, weights vary).
    Score > -0.3  → BULL
    Score < -0.5  → BEAR
    otherwise     → NEUTRAL
    """

    FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    CACHE_TTL = 3600  # 1 hour for regime data

    def __init__(self, cache_dir: str = '../cache/v3', fred_api_key: str = None):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.fred_api_key = fred_api_key
        self._regime = None
        self._score = None
        self._components = {}

    # ── FRED data fetch ───────────────────────────────────────────────────────
    def _fetch_fred(self, series_id: str) -> float | None:
        """Fetch the latest value of a FRED series. Falls back to cached value."""
        cache_file = os.path.join(self.cache_dir, f'fred_{series_id}.json')

        # Return cached value if fresh
        if os.path.exists(cache_file):
            age = time.time() - os.path.getmtime(cache_file)
            if age < self.CACHE_TTL:
                with open(cache_file) as f:
                    return json.load(f).get('value')

        if not HAS_REQUESTS:
            # If requests not available, try stale cache
            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    return json.load(f).get('value')
            return None

        url = f"{self.FRED_BASE}{series_id}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            lines = resp.text.strip().split('\n')
            # Last non-empty line
            for line in reversed(lines):
                parts = line.strip().split(',')
                if len(parts) == 2 and parts[1] not in ('.', ''):
                    try:
                        val = float(parts[1])
                        with open(cache_file, 'w') as f:
                            json.dump({'value': val, 'date': parts[0]}, f)
                        return val
                    except ValueError:
                        continue
        except Exception:
            pass

        # Stale cache fallback
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f).get('value')
        return None

    # ── Component reads ───────────────────────────────────────────────────────
    def _vix(self) -> float:
        try:
            vix = yf.Ticker('^VIX').history(period='5d', progress=False)['Close']
            return float(vix.iloc[-1]) if len(vix) > 0 else 20.0
        except Exception:
            return 20.0

    def _yield_curve(self) -> float | None:
        return self._fetch_fred('T10Y2Y')

    def _credit_spread(self) -> float | None:
        return self._fetch_fred('BAMLH0A0HYM2')

    def _spy_vs_sma(self) -> tuple[float, float]:
        try:
            spy = yf.download('SPY', period='1y', progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy = spy.droplevel(1, axis=1)
            close = spy['Close'].dropna()
            sma200 = float(close.rolling(200).mean().iloc[-1])
            current = float(close.iloc[-1])
            return current, sma200
        except Exception:
            return 1.0, 1.0

    # ── Main detect ───────────────────────────────────────────────────────────
    def detect(self, verbose: bool = True) -> str:
        """
        Detect regime. Returns 'BULL', 'NEUTRAL', or 'BEAR'.
        Results cached on the instance for the duration of the session.
        """
        vix_val       = self._vix()
        ycurve        = self._yield_curve()
        credit        = self._credit_spread()
        spy_px, spy_sma = self._spy_vs_sma()

        # Bear signals (each contributes negatively to score)
        vix_bear    = vix_val > 25
        curve_inv   = (ycurve is not None) and (ycurve < 0)
        spread_wide = (credit is not None) and (credit > 5.0)
        spy_below   = spy_px < spy_sma

        # Weighted score: heavier weight to yield curve + credit spread
        score = (
            -0.25 * float(vix_bear)
            - 0.35 * float(curve_inv)
            - 0.25 * float(spread_wide)
            - 0.15 * float(spy_below)
        )

        if score > -0.25:
            regime = 'BULL'
        elif score < -0.50:
            regime = 'BEAR'
        else:
            regime = 'NEUTRAL'

        self._regime = regime
        self._score  = score
        self._components = {
            'vix':          round(vix_val, 2),
            'yield_curve':  round(ycurve, 3)  if ycurve  is not None else None,
            'credit_spread': round(credit, 3) if credit  is not None else None,
            'spy_price':    round(spy_px, 2),
            'spy_sma200':   round(spy_sma, 2),
            'bear_signals': {
                'vix_high':     bool(vix_bear),
                'curve_inv':    bool(curve_inv),
                'spread_wide':  bool(spread_wide),
                'spy_below_sma': bool(spy_below),
            },
            'score':  round(score, 3),
            'regime': regime,
        }

        if verbose:
            self._print_summary()

        return regime

    def _print_summary(self):
        c = self._components
        print("=" * 50)
        print("  MACRO REGIME DETECTOR")
        print("=" * 50)
        print(f"  VIX           : {c['vix']:>7.2f}   {'⚠️  HIGH' if c['bear_signals']['vix_high'] else '✅  OK'}")
        yc = f"{c['yield_curve']:>+7.3f}%" if c['yield_curve'] is not None else "    N/A"
        print(f"  Yield Curve   : {yc}   {'⚠️  INVERTED' if c['bear_signals']['curve_inv'] else '✅  NORMAL'}")
        cs = f"{c['credit_spread']:>7.3f}%" if c['credit_spread'] is not None else "    N/A"
        print(f"  Credit Spread : {cs}   {'⚠️  WIDE' if c['bear_signals']['spread_wide'] else '✅  OK'}")
        print(f"  SPY vs SMA200 : {c['spy_price']:>7.2f} vs {c['spy_sma200']:.2f}   "
              f"{'⚠️  BELOW' if c['bear_signals']['spy_below_sma'] else '✅  ABOVE'}")
        print(f"  Regime Score  : {c['score']:>+.3f}")
        emoji = {'BULL': '🟢', 'NEUTRAL': '🟡', 'BEAR': '🔴'}[self._regime]
        print(f"  Regime        :  {emoji}  {self._regime}")
        print("=" * 50)

    # ── Accessors ─────────────────────────────────────────────────────────────
    def get_basis_weights(self) -> dict:
        """Return dynamic quantum basis weights for the detected regime."""
        if self._regime is None:
            self.detect(verbose=False)
        return REGIME_WEIGHTS[self._regime]

    def get_components(self) -> dict:
        """Return all raw indicator values and bear-signal flags."""
        if self._regime is None:
            self.detect(verbose=False)
        return self._components

    def regime(self) -> str:
        if self._regime is None:
            self.detect(verbose=False)
        return self._regime
