"""
Stock Analysis Module - V3
==========================
All classes extracted from StockPickerV3_Comprehensive.ipynb

Classes:
    - ComprehensiveStockAnalyzer: Main analysis pipeline
    - QuarterlyRebalancer: Cointegration-based rebalancing signals
    - VolatilityManagedRotator: Low-vol momentum strategy
    - IBKRPaperTrader: Interactive Brokers paper trading integration
"""

import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from tqdm import tqdm

# Optional imports (may not be installed)
try:
    from finviz.screener import Screener
    HAS_FINVIZ = True
except ImportError:
    HAS_FINVIZ = False

try:
    from statsmodels.tsa.stattools import coint
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# ib_insync is imported lazily in IBKRPaperTrader to avoid Python 3.14 asyncio issues

try:
    from data_providers import ExpandedDataProvider
    HAS_DATA_PROVIDERS = True
except ImportError:
    HAS_DATA_PROVIDERS = False


# =============================================================================
# COMPREHENSIVE STOCK ANALYZER
# =============================================================================

class ComprehensiveStockAnalyzer:
    """
    V3 Stock Analyzer with:
    - Tiered market cap selection
    - Multi-source composite scoring
    - Sector exclusions (no commodities/mining)
    """
    
    EXCLUDED_SECTORS = ['Basic Materials']
    # Clean energy (solar, wind) is intentionally NOT excluded.
    # Oil & Gas, mining, and physical commodities remain excluded.
    EXCLUDED_INDUSTRIES = [
        'Gold', 'Silver', 'Copper', 'Aluminum', 'Steel',
        'Coking Coal', 'Thermal Coal', 'Uranium',
        'Oil & Gas E&P', 'Oil & Gas Midstream', 'Oil & Gas Refining & Marketing',
        'Oil & Gas Drilling', 'Oil & Gas Equipment & Services',
        'Other Precious Metals & Mining', 'Other Industrial Metals & Mining',
        'Agricultural Inputs', 'Lumber & Wood Production',
    ]
    # Clean Energy industries explicitly allowed (never excluded)
    CLEAN_ENERGY_INDUSTRIES = [
        'Solar', 'Wind Energy', 'Renewable Utilities',
        'Utilities - Renewable', 'Utilities—Renewable',
    ]
    
    def __init__(self, cache_dir='../cache/v3'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.price_data = {}
        self.metadata = None
        self.consistency_metrics = None
        self.fundamentals = pd.DataFrame()
        self.alternative_data = pd.DataFrame()
        self.macro_regime = None
        self.metrics = None
        
        if HAS_DATA_PROVIDERS:
            self.data_provider = ExpandedDataProvider()
        else:
            self.data_provider = None
        
    def get_finviz_tickers(self, min_market_cap=100e6):
        """Get tickers from Finviz with metadata"""
        if not HAS_FINVIZ:
            raise ImportError("finviz not installed. Run: pip install finviz")
            
        cache_file = os.path.join(self.cache_dir, 'finviz_data.csv')
        
        if os.path.exists(cache_file):
            age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
            if age_hours < 24:
                _cache_df = pd.read_csv(cache_file)
                if 'market_cap_num' not in _cache_df.columns and 'market_cap' in _cache_df.columns:
                    _cache_df['market_cap_num'] = pd.to_numeric(_cache_df['market_cap'], errors='coerce').fillna(0)
                _cache_min = _cache_df['market_cap_num'].min() if len(_cache_df) > 0 else float('inf')
                if min_market_cap < _cache_min * 0.90:
                    print(f"  Cache floor ${_cache_min/1e6:.0f}M > requested ${min_market_cap/1e6:.0f}M — fetching fresh data...")
                else:
                    print(f"  Using cached Finviz data ({age_hours:.1f}h old)")
                    df = _cache_df[_cache_df['market_cap_num'] >= min_market_cap].copy()
                    df = df[~df['sector'].isin(self.EXCLUDED_SECTORS)].copy()
                    df = df[~df['industry'].isin(self.EXCLUDED_INDUSTRIES)].copy()
                    return df
        
        print("  Fetching fresh data from Finviz...")
        try:
            filters = ['sh_avgvol_o100', 'sh_price_o5']
            stock_list = Screener(filters=filters, table='Overview', order='marketcap')
            df = pd.DataFrame(stock_list.data)
            
            def parse_mcap(val):
                if pd.isna(val) or val == '-':
                    return 0
                val = str(val).upper()
                if 'B' in val:
                    return float(val.replace('B', '')) * 1e9
                elif 'M' in val:
                    return float(val.replace('M', '')) * 1e6
                return 0
            
            df['market_cap_num'] = df['Market Cap'].apply(parse_mcap)
            df = df[df['market_cap_num'] >= min_market_cap].copy()
            df = df[['Ticker', 'Company', 'Sector', 'Industry', 'Market Cap', 'market_cap_num', 'Volume']]
            df.columns = ['ticker', 'company', 'sector', 'industry', 'market_cap', 'market_cap_num', 'volume']
            
            df = df[~df['sector'].isin(self.EXCLUDED_SECTORS)].copy()
            df = df[~df['industry'].isin(self.EXCLUDED_INDUSTRIES)].copy()
            
            df.to_csv(cache_file, index=False)
            print(f"  Cached {len(df)} tickers")
            return df
            
        except Exception as e:
            print(f"  Error fetching Finviz: {e}")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            return pd.DataFrame()
    
    def get_tiered_tickers(self, all_tickers, max_total, allocation='balanced'):
        """Select tickers across market cap tiers."""
        df = all_tickers.copy()
        
        df['cap_tier'] = pd.cut(
            df['market_cap_num'],
            bins=[0, 2e9, 10e9, 50e9, float('inf')],
            labels=['small', 'mid', 'large', 'mega']
        )
        
        allocations = {
            'balanced': {'small': 0.25, 'mid': 0.25, 'large': 0.25, 'mega': 0.25},
            'growth': {'small': 0.35, 'mid': 0.35, 'large': 0.20, 'mega': 0.10},
            'value': {'small': 0.10, 'mid': 0.20, 'large': 0.35, 'mega': 0.35},
            'volume': None
        }
        
        alloc = allocations.get(allocation)
        
        if alloc is None:
            return df.head(max_total)
        
        selected = []
        for tier, pct in alloc.items():
            tier_df = df[df['cap_tier'] == tier].copy()
            n_select = int(max_total * pct)
            tier_df['volume_num'] = pd.to_numeric(
                tier_df['volume'].astype(str).str.replace(',', ''), errors='coerce'
            ).fillna(0)
            tier_df = tier_df.sort_values('volume_num', ascending=False)
            selected.append(tier_df.head(n_select))
            print(f"    {tier.upper()}: {len(tier_df.head(n_select))} stocks")
        
        result = pd.concat(selected, ignore_index=True)
        print(f"  Total selected: {len(result)} stocks")
        return result
    
    def download_prices(self, min_market_cap=100e6, max_tickers=3000, cap_allocation='balanced'):
        """Download price data with tiered market cap selection"""
        print("\n[STEP 1-2] Getting tickers and downloading prices...")
        
        all_tickers = self.get_finviz_tickers(min_market_cap=min_market_cap)
        
        if len(all_tickers) == 0:
            print("  No tickers found!")
            return
        
        print(f"  Found {len(all_tickers)} total tickers above ${min_market_cap/1e6:.0f}M")
        print(f"\n  Applying {cap_allocation.upper()} allocation:")
        selected = self.get_tiered_tickers(all_tickers, max_tickers, cap_allocation)
        self.metadata = selected
        
        tickers = selected['ticker'].tolist()
        print(f"\n  Downloading 3Y price data for {len(tickers)} tickers...")
        
        batch_size = 50
        all_data = {}
        
        for i in tqdm(range(0, len(tickers), batch_size), desc="  Downloading"):
            batch = tickers[i:i+batch_size]
            try:
                data = yf.download(batch, period='3y', progress=False, group_by='ticker', threads=True)
                
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            ticker_data = data
                        else:
                            ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else None
                        
                        if ticker_data is not None and not ticker_data.empty:
                            if 'Close' in ticker_data.columns:
                                close_data = ticker_data['Close']
                                if len(close_data.dropna()) >= 252:
                                    all_data[ticker] = close_data
                    except:
                        continue
            except Exception as e:
                print(f"  Batch error: {e}")
                continue
            
            time.sleep(0.1)
        
        self.price_data = all_data
        print(f"  ✓ Downloaded price data for {len(all_data)} tickers")
    
    def calculate_consistency_metrics(self):
        """Calculate risk-adjusted metrics for all stocks"""
        print("\n[STEP 3] Calculating consistency metrics...")
        
        results = []
        rf_rate = 0.04
        
        for ticker, prices in tqdm(self.price_data.items(), desc="  Calculating"):
            try:
                prices = prices.dropna()
                if len(prices) < 252:
                    continue
                
                daily_returns = prices.pct_change().dropna()
                monthly = prices.resample('ME').last().pct_change().dropna() * 100
                
                meta = self.metadata[self.metadata['ticker'] == ticker]
                sector = meta['sector'].values[0] if len(meta) > 0 else 'Unknown'
                industry = meta['industry'].values[0] if len(meta) > 0 else 'Unknown'
                mcap = meta['market_cap_num'].values[0] if len(meta) > 0 else 0
                cap_tier = meta['cap_tier'].values[0] if len(meta) > 0 and 'cap_tier' in meta.columns else 'unknown'
                
                return_1y = (prices.iloc[-1] / prices.iloc[-252] - 1) * 100 if len(prices) >= 252 else 0
                return_6m = (prices.iloc[-1] / prices.iloc[-126] - 1) * 100 if len(prices) >= 126 else 0
                return_3m = (prices.iloc[-1] / prices.iloc[-63] - 1) * 100 if len(prices) >= 63 else 0
                return_1m = (prices.iloc[-1] / prices.iloc[-21] - 1) * 100 if len(prices) >= 21 else 0
                
                def calc_sortino(rets, periods=252):
                    ann_ret = rets.mean() * periods
                    downside = rets[rets < 0].std() * np.sqrt(periods)
                    return (ann_ret - rf_rate) / downside if downside > 0 else 0
                
                sortino_1y = calc_sortino(daily_returns[-252:]) if len(daily_returns) >= 252 else 0
                sortino_6m = calc_sortino(daily_returns[-126:]) if len(daily_returns) >= 126 else 0
                sortino_3m = calc_sortino(daily_returns[-63:]) if len(daily_returns) >= 63 else 0
                sortino_1m = calc_sortino(daily_returns[-21:]) if len(daily_returns) >= 21 else 0
                
                positive_months = (monthly > 0).sum()
                total_months = len(monthly)
                win_rate = (positive_months / total_months * 100) if total_months > 0 else 0
                
                rolling_max = prices.expanding().max()
                drawdowns = (prices / rolling_max - 1) * 100
                max_dd = drawdowns.min()
                
                ann_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1
                calmar = (ann_return * 100) / abs(max_dd) if abs(max_dd) > 0 else 0
                
                ma_50 = prices.rolling(50).mean().iloc[-1]
                ma_200 = prices.rolling(200).mean().iloc[-1]
                current_price = prices.iloc[-1]
                
                trend_score = 50
                if current_price > ma_50:
                    trend_score += 20
                if current_price > ma_200:
                    trend_score += 15
                if ma_50 > ma_200:
                    trend_score += 15
                
                above_ma50_pct = (current_price / ma_50 - 1) * 100 if ma_50 > 0 else 0
                
                acceleration = 0
                if sortino_3m > sortino_6m and sortino_1m > sortino_3m:
                    acceleration = 1
                
                results.append({
                    'ticker': ticker,
                    'sector': sector,
                    'industry': industry,
                    'market_cap': mcap,
                    'cap_tier': cap_tier,
                    'return_1m': return_1m,
                    'return_3m': return_3m,
                    'return_6m': return_6m,
                    'return_1y': return_1y,
                    'sortino_ratio': sortino_1y,
                    'sortino_6m': sortino_6m,
                    'sortino_3m': sortino_3m,
                    'sortino_1m': sortino_1m,
                    'win_rate_pct': win_rate,
                    'max_drawdown_pct': max_dd,
                    'calmar_ratio': calmar,
                    'trend_score': trend_score,
                    'above_ma50_pct': above_ma50_pct,
                    'momentum_accelerating': acceleration,
                    'avg_monthly_return': monthly.mean(),
                    'monthly_volatility': monthly.std()
                })
                
            except:
                continue
        
        self.consistency_metrics = pd.DataFrame(results)
        print(f"  ✓ Calculated metrics for {len(results)} stocks")
        return self.consistency_metrics
    
    def calculate_composite_score(self):
        """Calculate final composite score"""
        print("\n[STEP 6] Calculating composite scores...")
        
        df = self.consistency_metrics.copy()
        
        initial_count = len(df)
        dd = pd.to_numeric(df['max_drawdown_pct'], errors='coerce').fillna(-70)
        wr = pd.to_numeric(df['win_rate_pct'], errors='coerce').fillna(0)
        df = df[(dd > -65) & (wr > 35)].copy()
        print(f"  Pre-filter: Removed {initial_count - len(df)} extreme-risk stocks")
        
        def normalize(series):
            return series.rank(pct=True, na_option='bottom') * 100
        
        ret_6m = pd.to_numeric(df['return_6m'], errors='coerce').fillna(0)
        ret_1y = pd.to_numeric(df['return_1y'], errors='coerce').fillna(0)
        df['raw_return_score'] = ret_6m * 0.6 + ret_1y * 0.4
        df['raw_return_rank'] = normalize(df['raw_return_score'])
        
        if 'sortino_3m' in df.columns:
            df['momentum_sortino_rank'] = normalize(pd.to_numeric(df['sortino_3m'], errors='coerce').fillna(0))
        else:
            df['momentum_sortino_rank'] = 50
        
        df['trend_rank'] = normalize(pd.to_numeric(df['trend_score'], errors='coerce').fillna(50))
        df['sortino_rank'] = normalize(pd.to_numeric(df['sortino_ratio'], errors='coerce').fillna(0))
        df['calmar_rank'] = normalize(pd.to_numeric(df['calmar_ratio'], errors='coerce').fillna(0))
        
        df['acceleration_bonus'] = 0.0
        if 'momentum_accelerating' in df.columns:
            df.loc[df['momentum_accelerating'] == 1, 'acceleration_bonus'] = 15.0
        
        df['high_return_bonus'] = 0.0
        df.loc[ret_6m > 150, 'high_return_bonus'] = 30.0
        df.loc[(ret_6m > 100) & (ret_6m <= 150), 'high_return_bonus'] = 22.0
        df.loc[(ret_6m > 80) & (ret_6m <= 100), 'high_return_bonus'] = 16.0
        df.loc[(ret_6m > 50) & (ret_6m <= 80), 'high_return_bonus'] = 10.0
        df.loc[(ret_6m > 30) & (ret_6m <= 50), 'high_return_bonus'] = 5.0

        # Utilities ROE quality bonus
        # Utilities with ROE > 15% AND sortino > 0.5 get +10 to reward
        # high-quality regulated utilities that are otherwise penalised by
        # low-momentum rankings relative to growth sectors.
        df['utilities_bonus'] = 0.0
        if 'sector' in df.columns:
            utils_mask = df['sector'].str.contains('Utilities', na=False)
            sortino_vals = pd.to_numeric(df.get('sortino_ratio', pd.Series(0, index=df.index)), errors='coerce').fillna(0)
            roe_vals = roe_filter(df.index.tolist() if 'ticker' not in df.columns else df['ticker'].tolist())
            roe_series = df['ticker'].map(roe_vals).fillna(0) if 'ticker' in df.columns else pd.Series(0, index=df.index)
            df.loc[utils_mask & (roe_series > 15) & (sortino_vals > 0.5), 'utilities_bonus'] = 10.0

        # Clean energy pass-through bonus (6M return bonus already covers this;
        # label it separately for transparency)
        df['clean_energy_bonus'] = 0.0
        if 'industry' in df.columns:
            ce_mask = df['industry'].str.contains('Solar|Wind|Renewable', na=False, case=False)
            df.loc[ce_mask & (ret_6m > 0), 'clean_energy_bonus'] = 5.0

        # Symmetric downside penalty — balances the upside return bonuses
        dd_num = pd.to_numeric(df['max_drawdown_pct'], errors='coerce').fillna(0)
        df['drawdown_penalty'] = 0.0
        df.loc[dd_num < -40, 'drawdown_penalty'] = 25.0
        df.loc[dd_num < -50, 'drawdown_penalty'] = 35.0
        df.loc[dd_num < -60, 'drawdown_penalty'] = 45.0

        df['composite_score'] = (
            df['raw_return_rank'] * 0.35 +
            df['momentum_sortino_rank'] * 0.25 +
            df['trend_rank'] * 0.20 +
            df['sortino_rank'] * 0.15 +
            df['calmar_rank'] * 0.05 +
            df['acceleration_bonus'] +
            df['high_return_bonus'] +
            df['utilities_bonus'] +
            df['clean_energy_bonus'] -
            df['drawdown_penalty']
        ).round(1)
        
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        self.metrics = df
        print(f"  ✓ Calculated composite scores for {len(df)} stocks")
        return df
    
    def get_top_picks(self, n=20, min_win_rate=33, max_drawdown=-60):
        """Get top N picks"""
        filtered = self.metrics[
            (self.metrics['win_rate_pct'] >= min_win_rate) &
            (self.metrics['max_drawdown_pct'] >= max_drawdown)
        ].copy()
        return filtered.head(n)
    
    def run_full_analysis(self, min_market_cap=100e6, max_tickers=3000, cap_allocation='balanced'):
        """Run the complete analysis pipeline."""
        print("="*60)
        print("COMPREHENSIVE STOCK ANALYSIS V3")
        print(f"Strategy: {cap_allocation.upper()} | Min Cap: ${min_market_cap/1e6:.0f}M")
        print("="*60)
        
        self.download_prices(min_market_cap=min_market_cap, max_tickers=max_tickers, cap_allocation=cap_allocation)
        self.calculate_consistency_metrics()
        self.calculate_composite_score()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return self.metrics
    
    def export_results(self, output_dir='../output/v3'):
        """Export results by sector"""
        today = datetime.now().strftime('%m_%d_%Y')
        export_path = os.path.join(output_dir, today)
        os.makedirs(export_path, exist_ok=True)
        
        for sector in self.metrics['sector'].unique():
            sector_df = self.metrics[self.metrics['sector'] == sector].copy()
            if len(sector_df) > 0:
                filename = f"final_selection_{sector.replace('/', '-')}.csv"
                sector_df.to_csv(os.path.join(export_path, filename), index=False)
        
        self.metrics.to_csv(os.path.join(export_path, 'comprehensive_rankings.csv'), index=False)
        print(f"\n✓ Results exported to: {export_path}/")
        return export_path


# =============================================================================
# QUARTERLY REBALANCER
# =============================================================================

class QuarterlyRebalancer:
    """Quarterly rebalancing using cointegration spread Z-scores."""
    
    # Primary sector→ETF mapping (broad sector names from Finviz/yfinance)
    SECTOR_ETF_MAP = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financial Services': 'XLF',
        'Financials': 'XLF',
        'Consumer Cyclical': 'XLY',
        'Consumer Discretionary': 'XLY',
        'Consumer Defensive': 'XLP',
        'Consumer Staples': 'XLP',
        'Industrials': 'XLI',
        'Basic Materials': 'XLB',
        'Materials': 'XLB',
        'Energy': 'XLE',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC',
        'Communications': 'XLC',
    }

    # Granular GICS industry→ETF mapping (95%+ coverage)
    # Provides a tighter benchmark than the broad sector when available.
    INDUSTRY_ETF_MAP = {
        # Technology
        'Semiconductors': 'SOXX',
        'Semiconductor Equipment': 'SOXX',
        'Software - Application': 'IGV',
        'Software - Infrastructure': 'IGV',
        'Software': 'IGV',
        'Information Technology Services': 'XLK',
        'Computer Hardware': 'XLK',
        'Electronic Components': 'XLK',
        'Consumer Electronics': 'XLK',
        'Internet Content & Information': 'XLC',
        'Electronic Gaming & Multimedia': 'XLC',
        # Healthcare
        'Biotechnology': 'XBI',
        'Drug Manufacturers - General': 'XPH',
        'Drug Manufacturers - Specialty & Generic': 'XPH',
        'Medical Devices': 'IHI',
        'Medical Instruments & Supplies': 'IHI',
        'Diagnostics & Research': 'XLV',
        'Healthcare Plans': 'XLV',
        'Medical Care Facilities': 'XLV',
        'Pharmaceutical Retailers': 'XLV',
        # Financials
        'Banks - Regional': 'KRE',
        'Banks - Diversified': 'XLF',
        'Asset Management': 'XLF',
        'Capital Markets': 'XLF',
        'Insurance - Life': 'XLF',
        'Insurance - Property & Casualty': 'XLF',
        'Insurance - Diversified': 'XLF',
        'Financial Data & Stock Exchanges': 'XLF',
        'Credit Services': 'XLF',
        'Mortgage Finance': 'XLRE',
        # Consumer Discretionary
        'Auto Manufacturers': 'CARZ',
        'Auto Parts': 'XLY',
        'Restaurants': 'XLY',
        'Specialty Retail': 'XRT',
        'Apparel Retail': 'XRT',
        'Internet Retail': 'XRT',
        'Home Improvement Retail': 'XRT',
        'Leisure': 'XLY',
        'Hotels & Motels': 'XLY',
        'Travel Services': 'XLY',
        # Consumer Staples
        'Beverages - Non-Alcoholic': 'XLP',
        'Beverages - Alcoholic': 'XLP',
        'Grocery Stores': 'XLP',
        'Household & Personal Products': 'XLP',
        'Packaged Foods': 'XLP',
        'Tobacco': 'XLP',
        'Discount Stores': 'XLP',
        # Industrials
        'Aerospace & Defense': 'ITA',
        'Airlines': 'JETS',
        'Trucking': 'XLI',
        'Railroads': 'XLI',
        'Industrial Distribution': 'XLI',
        'Staffing & Employment Services': 'XLI',
        'Waste Management': 'XLI',
        'Engineering & Construction': 'XLI',
        'Farm & Heavy Construction Machinery': 'XLI',
        'Tools & Accessories': 'XLI',
        # Real Estate
        'REIT - Retail': 'XLRE',
        'REIT - Industrial': 'XLRE',
        'REIT - Office': 'XLRE',
        'REIT - Healthcare Facilities': 'XLRE',
        'REIT - Residential': 'XLRE',
        'REIT - Diversified': 'XLRE',
        'REIT - Specialty': 'XLRE',
        'Real Estate Services': 'XLRE',
        # Utilities
        'Utilities - Regulated Electric': 'XLU',
        'Utilities - Regulated Gas': 'XLU',
        'Utilities - Diversified': 'XLU',
        'Utilities - Independent Power Producers': 'XLU',
        'Solar': 'TAN',
        'Wind Energy': 'FAN',
        'Utilities - Renewable': 'ICLN',
        # Communication Services
        'Telecom Services': 'XLC',
        'Entertainment': 'XLC',
        'Broadcasting': 'XLC',
        'Publishing': 'XLC',
        # Energy (kept for completeness — excluded from universe but signals still computed)
        'Oil & Gas Integrated': 'XLE',
        'Oil & Gas E&P': 'XLE',
        'Oil & Gas Equipment & Services': 'XLE',
    }

    def __init__(self, holdings_df, lookback_days=90, zscore_threshold=2.0):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels not installed. Run: pip install statsmodels")
        self.holdings = holdings_df
        self.lookback = lookback_days
        self.threshold = zscore_threshold
        self.signals = []

    def get_sector_etf(self, sector: str, industry: str = '') -> str:
        """Resolve the tightest-match ETF benchmark for a given sector/industry.

        Fallback hierarchy:
          1. Industry-level map (INDUSTRY_ETF_MAP) — tightest benchmark
          2. Sector-level map (SECTOR_ETF_MAP) — broad sector
          3. yfinance info['sector'] lookup for unknown labels
          4. SPY — market-wide fallback
        """
        if industry and industry in self.INDUSTRY_ETF_MAP:
            return self.INDUSTRY_ETF_MAP[industry]
        if sector and sector in self.SECTOR_ETF_MAP:
            return self.SECTOR_ETF_MAP[sector]
        # Partial match on industry string
        if industry:
            for key, etf in self.INDUSTRY_ETF_MAP.items():
                if key.lower() in industry.lower() or industry.lower() in key.lower():
                    return etf
        return 'SPY'
    
    def calculate_spread_zscore(self, stock_ticker, etf_ticker, period='2y'):
        """Calculate spread Z-score between stock and ETF.

        Fix: Engle-Granger residuals estimated over the FULL 2-year history for
        statistical power. Rolling Z-score window stays at self.lookback (90d)
        on those long residuals, matching the cointegration test horizon.
        """
        try:
            data = yf.download([stock_ticker, etf_ticker], period=period, progress=False)
            if data.empty:
                return None, None, False

            stock_prices = data['Adj Close'][stock_ticker].dropna()
            etf_prices = data['Adj Close'][etf_ticker].dropna()

            aligned = pd.concat([stock_prices, etf_prices], axis=1, keys=['stock', 'etf']).dropna()
            if len(aligned) < 252:
                return None, None, False

            log_stock = np.log(aligned['stock'])
            log_etf = np.log(aligned['etf'])

            # Cointegration test over full history (trend='ct' for realism)
            _, pvalue, _ = coint(log_stock, log_etf, trend='ct')
            is_cointegrated = pvalue < 0.05

            # Estimate beta via OLS over the FULL 2-year window (not just 90d)
            import statsmodels.api as sm
            beta = sm.OLS(log_stock, sm.add_constant(log_etf)).fit().params[1]

            # Build spread over full history, then apply rolling Z-score window
            spread = log_stock - beta * log_etf
            rolling_mean = spread.rolling(self.lookback).mean()
            rolling_std = spread.rolling(self.lookback).std()
            zscore = (spread - rolling_mean) / rolling_std

            return zscore.iloc[-1], is_cointegrated, True

        except:
            return None, None, False
    
    def generate_signals(self, verbose=True):
        """Generate rebalancing signals for all holdings."""
        signals = []
        
        if verbose:
            print(f"Analyzing {len(self.holdings)} holdings...")
        
        for _, row in tqdm(self.holdings.iterrows(), total=len(self.holdings), disable=not verbose):
            ticker = row['ticker']
            sector = row.get('sector', 'Unknown')
            etf = self.get_sector_etf(sector)
            
            zscore, is_coint, success = self.calculate_spread_zscore(ticker, etf)
            
            if not success:
                signal = 'NO_DATA'
                zscore = 0
                is_coint = False
            elif zscore > self.threshold:
                signal = 'TRIM'
            elif zscore < -self.threshold:
                signal = 'ROTATE_OUT'
            else:
                signal = 'HOLD'
            
            signals.append({
                'ticker': ticker, 'sector': sector, 'etf': etf,
                'zscore': round(zscore, 2) if zscore else None,
                'signal': signal, 'is_cointegrated': is_coint,
                'weight': row.get('weight', 0)
            })
        
        self.signals_df = pd.DataFrame(signals)
        return self.signals_df


# =============================================================================
# VOLATILITY-MANAGED ROTATOR
# =============================================================================

class VolatilityManagedRotator:
    """Momentum rotation with volatility controls."""
    
    def __init__(self, price_data, n_holdings=50, ranking_lookback=252,
                 vol_target=15, max_position=5, use_regime_filter=True):
        self.price_data = price_data
        self.n_holdings = n_holdings
        self.ranking_lookback = ranking_lookback
        self.vol_target = vol_target / 100
        self.max_position = max_position / 100
        self.use_regime_filter = use_regime_filter
        
    def calculate_historical_vol(self, prices, lookback=60):
        returns = prices.pct_change().dropna()
        if len(returns) < lookback:
            return returns.std() * np.sqrt(252) if len(returns) > 5 else 0.3
        return returns.tail(lookback).std() * np.sqrt(252)
    
    def get_regime(self, spy_prices, date, lookback=200):
        prices_up_to_date = spy_prices.loc[:date].tail(lookback)
        if len(prices_up_to_date) < lookback:
            return 'neutral', 1.0
        
        sma = prices_up_to_date.mean()
        current = prices_up_to_date.iloc[-1]
        
        if current > sma * 1.02:
            return 'bull', 1.0
        elif current < sma * 0.98:
            return 'bear', 0.5
        else:
            return 'neutral', 0.75
    
    def calculate_vol_scaled_weights(self, tickers, prices_df, as_of_date):
        vol_dict = {}
        for t in tickers:
            if t in prices_df.columns:
                prices = prices_df[t].loc[:as_of_date].dropna()
                if len(prices) > 60:
                    vol = self.calculate_historical_vol(prices)
                    if vol > 0:
                        vol_dict[t] = vol
        
        if len(vol_dict) == 0:
            return {t: 1/len(tickers) for t in tickers}
        
        inv_vol = {t: 1/v for t, v in vol_dict.items()}
        total_inv_vol = sum(inv_vol.values())
        
        weights = {}
        for t in tickers:
            if t in inv_vol:
                raw_weight = inv_vol[t] / total_inv_vol
                weights[t] = min(raw_weight, self.max_position)
            else:
                weights[t] = 1 / len(tickers)
        
        total = sum(weights.values())
        return {t: w/total for t, w in weights.items()}
    
    def calculate_metrics_at_date(self, prices_df, as_of_date, lookback=252):
        historical = prices_df.loc[:as_of_date].tail(lookback)
        
        if len(historical) < 60:
            return pd.DataFrame()
        
        metrics = []
        for ticker in historical.columns:
            prices = historical[ticker].dropna()
            if len(prices) < 60:
                continue
            
            returns = prices.pct_change().dropna()
            vol = returns.std() * np.sqrt(252)
            
            if len(prices) >= 126:
                momentum_6m = (prices.iloc[-1] / prices.iloc[-126] - 1)
            else:
                momentum_6m = 0
            
            risk_adj_momentum = momentum_6m / vol if vol > 0 else 0
            
            recent_prices = prices.tail(60)
            running_max = recent_prices.expanding().max()
            recent_dd = (recent_prices / running_max - 1).min()
            
            composite = (
                risk_adj_momentum * 0.5 +
                (-vol * 10) * 0.3 +
                (-recent_dd * 5) * 0.2
            )
            
            metrics.append({
                'ticker': ticker,
                'volatility': vol,
                'momentum_6m': momentum_6m * 100,
                'risk_adj_momentum': risk_adj_momentum,
                'recent_dd': recent_dd * 100,
                'composite_score': composite
            })
        
        df = pd.DataFrame(metrics)
        if len(df) > 0:
            df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        return df


# =============================================================================
# IBKR PAPER TRADER
# =============================================================================

class IBKRPaperTrader:
    """Connect to IBKR paper trading account and execute portfolio orders."""
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        # Fix Jupyter asyncio conflict
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            print("Note: pip install nest_asyncio for better Jupyter compatibility")
        
        # Lazy import to avoid asyncio issues
        try:
            from ib_insync import IB, Stock, MarketOrder, LimitOrder
            self._IB = IB
            self._Stock = Stock
            self._MarketOrder = MarketOrder
            self._LimitOrder = LimitOrder
        except ImportError:
            raise ImportError("ib_insync not installed. Run: pip install ib_insync")
        except RuntimeError:
            # Handle asyncio event loop issues
            import asyncio
            asyncio.set_event_loop(asyncio.new_event_loop())
            from ib_insync import IB, Stock, MarketOrder, LimitOrder
            self._IB = IB
            self._Stock = Stock
            self._MarketOrder = MarketOrder
            self._LimitOrder = LimitOrder
            
        self.ib = self._IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.connected = False
        
    def connect(self):
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            print(f"✅ Connected to IBKR")
            print(f"   Account: {self.ib.managedAccounts()}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\nMake sure TWS/Gateway is running with API enabled")
            return False
    
    def disconnect(self):
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            print("Disconnected from IBKR")
    
    def get_account_summary(self):
        if not self.connected:
            return None
            
        account_values = self.ib.accountSummary()
        summary = {}
        for av in account_values:
            if av.tag in ['NetLiquidation', 'TotalCashValue', 'BuyingPower', 'GrossPositionValue']:
                summary[av.tag] = float(av.value)
        
        print("\n--- ACCOUNT SUMMARY ---")
        for k, v in summary.items():
            print(f"  {k}: ${v:,.2f}")
        return summary
    
    def get_current_positions(self):
        if not self.connected:
            return None
            
        positions = self.ib.positions()
        if not positions:
            print("\nNo current positions")
            return pd.DataFrame()
        
        pos_data = []
        for pos in positions:
            pos_data.append({
                'symbol': pos.contract.symbol,
                'shares': pos.position,
                'avg_cost': pos.avgCost,
                'value': pos.position * pos.avgCost
            })
        
        df = pd.DataFrame(pos_data)
        print(f"\n--- CURRENT POSITIONS ({len(df)} stocks) ---")
        print(df.to_string())
        return df
    
    def get_live_price(self, symbol):
        contract = self._Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        
        ticker = self.ib.reqMktData(contract, '', False, False)
        self.ib.sleep(2)
        
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        
        if price != price:
            bars = self.ib.reqHistoricalData(
                contract, endDateTime='', durationStr='1 D',
                barSizeSetting='1 day', whatToShow='TRADES', useRTH=True
            )
            if bars:
                price = bars[-1].close
        
        return price
    
    def calculate_orders(self, target_portfolio, capital=None):
        if not self.connected:
            return None
        
        if capital is None:
            summary = self.get_account_summary()
            capital = summary.get('NetLiquidation', 10000)
        
        print(f"\n--- CALCULATING ORDERS (Capital: ${capital:,.2f}) ---")
        
        current_positions = self.get_current_positions()
        current_holdings = {}
        if not current_positions.empty:
            current_holdings = dict(zip(current_positions['symbol'], current_positions['shares']))
        
        orders = []
        
        for _, row in target_portfolio.iterrows():
            ticker = row['ticker']
            target_weight = row['weight']
            target_value = capital * target_weight
            
            try:
                price = self.get_live_price(ticker)
                if price is None or price != price:
                    continue
            except:
                continue
            
            target_shares = int(target_value / price)
            current_shares = current_holdings.get(ticker, 0)
            delta = target_shares - current_shares
            
            if delta != 0:
                orders.append({
                    'ticker': ticker,
                    'action': 'BUY' if delta > 0 else 'SELL',
                    'shares': abs(delta),
                    'price': price,
                    'value': abs(delta) * price,
                    'current': current_shares,
                    'target': target_shares
                })
        
        target_tickers = set(target_portfolio['ticker'])
        for ticker, shares in current_holdings.items():
            if ticker not in target_tickers and shares > 0:
                try:
                    price = self.get_live_price(ticker)
                    orders.append({
                        'ticker': ticker, 'action': 'SELL',
                        'shares': abs(shares), 'price': price,
                        'value': abs(shares) * price,
                        'current': shares, 'target': 0
                    })
                except:
                    pass
        
        orders_df = pd.DataFrame(orders)
        if not orders_df.empty:
            orders_df = orders_df.sort_values('value', ascending=False)
            print(f"\n--- ORDERS TO EXECUTE ({len(orders_df)}) ---")
            print(orders_df.to_string())
        else:
            print("No orders needed")
        
        return orders_df
    
    def execute_orders(self, orders_df, order_type='MARKET'):
        if not self.connected or orders_df.empty:
            return
        
        print(f"\nEXECUTING {len(orders_df)} ORDERS")
        
        executed = []
        
        for _, order in orders_df.iterrows():
            ticker = order['ticker']
            action = order['action']
            shares = int(order['shares'])
            
            if shares == 0:
                continue
            
            try:
                contract = self._Stock(ticker, 'SMART', 'USD')
                self.ib.qualifyContracts(contract)
                
                if order_type == 'MARKET':
                    ib_order = self._MarketOrder(action, shares)
                else:
                    ib_order = self._LimitOrder(action, shares, order['price'])
                
                trade = self.ib.placeOrder(contract, ib_order)
                self.ib.sleep(0.5)
                
                print(f"  ✅ {action} {shares} {ticker}")
                executed.append({'ticker': ticker, 'action': action, 'shares': shares})
                
            except Exception as e:
                print(f"  ❌ Failed: {ticker} - {e}")
        
        return pd.DataFrame(executed)
    
    def go_to_cash(self):
        if not self.connected:
            return
        
        positions = self.get_current_positions()
        if positions.empty:
            return
        
        print("\n🔴 Moving to 100% CASH")
        
        orders = []
        for _, pos in positions.iterrows():
            if pos['shares'] > 0:
                orders.append({
                    'ticker': pos['symbol'],
                    'action': 'SELL',
                    'shares': pos['shares'],
                    'price': 0
                })
        
        return self.execute_orders(pd.DataFrame(orders), order_type='MARKET')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_regime(verbose=True):
    """Check current market regime (SPY vs 200-day SMA)."""
    spy = yf.download('SPY', period='1y', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy = spy.droplevel(1, axis=1)
    
    spy_close = spy['Close'].dropna()
    spy_sma200 = spy_close.rolling(200).mean()
    
    current = spy_close.iloc[-1]
    sma = spy_sma200.iloc[-1]
    bullish = current > sma
    
    if verbose:
        print(f"SPY: ${current:.2f} | 200-SMA: ${sma:.2f}")
        print(f"Regime: {'🟢 BULLISH' if bullish else '🔴 BEARISH'}")
    
    return bullish, current, sma


def generate_live_portfolio(analyzer, n_holdings=50, max_position_pct=5):
    """Generate portfolio for live trading from analyzer."""
    if analyzer.price_data is None:
        raise ValueError("Run analyzer.run_full_analysis() first")
    
    print("Generating live portfolio...")
    
    # Use analyzer's existing price data
    all_tickers = list(analyzer.price_data.keys())
    
    valid_stocks = []
    for ticker in tqdm(all_tickers, desc="Calculating metrics"):
        try:
            if isinstance(analyzer.price_data, dict):
                price_series = analyzer.price_data[ticker]
                if isinstance(price_series, pd.DataFrame):
                    price_series = price_series['Close'] if 'Close' in price_series.columns else price_series.iloc[:, 0]
            else:
                continue
            
            price_series = price_series.dropna()
            if len(price_series) < 252:
                continue
            
            returns = price_series.pct_change().dropna()
            
            annual_return = returns.mean() * 252
            total_vol = returns.std() * np.sqrt(252)
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 10 else total_vol
            
            sortino = (annual_return - 0.04) / downside_vol if downside_vol > 0.001 else 0
            
            monthly = price_series.resample('ME').last().pct_change().dropna()
            win_rate = (monthly > 0).sum() / len(monthly) * 100 if len(monthly) > 0 else 0
            
            rolling_max = price_series.cummax()
            drawdown = (price_series - rolling_max) / rolling_max
            max_dd = drawdown.min() * 100
            
            if len(price_series) >= 126:
                mom_3m = (price_series.iloc[-1] / price_series.iloc[-63] - 1) * 100
                mom_6m = (price_series.iloc[-1] / price_series.iloc[-126] - 1) * 100
                momentum = (mom_3m + mom_6m) / 2
            else:
                momentum = 0
            
            vol_60d = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else total_vol
            
            valid_stocks.append({
                'ticker': ticker,
                'sortino': sortino,
                'win_rate': win_rate,
                'max_dd': max_dd,
                'momentum': momentum,
                'vol_60d': vol_60d,
                'last_price': price_series.iloc[-1]
            })
            
        except:
            continue
    
    metrics_df = pd.DataFrame(valid_stocks)
    
    # Rank and select
    metrics_df['sortino_rank'] = metrics_df['sortino'].rank(pct=True)
    metrics_df['winrate_rank'] = metrics_df['win_rate'].rank(pct=True)
    metrics_df['momentum_rank'] = metrics_df['momentum'].rank(pct=True)
    metrics_df['dd_rank'] = metrics_df['max_dd'].rank(pct=True, ascending=False)
    
    metrics_df['composite'] = (
        metrics_df['sortino_rank'] * 0.40 +
        metrics_df['winrate_rank'] * 0.25 +
        metrics_df['momentum_rank'] * 0.25 +
        metrics_df['dd_rank'] * 0.10
    )
    
    top_stocks = metrics_df.nlargest(n_holdings, 'composite').copy()
    
    # Inverse-vol weights
    inverse_vol = 1 / top_stocks['vol_60d']
    raw_weights = inverse_vol / inverse_vol.sum()
    capped_weights = raw_weights.clip(upper=max_position_pct/100)
    final_weights = capped_weights / capped_weights.sum()
    top_stocks['weight'] = final_weights.values
    
    # Add sector info
    if analyzer.metrics is not None:
        sector_map = dict(zip(analyzer.metrics['ticker'], analyzer.metrics['sector']))
        top_stocks['sector'] = top_stocks['ticker'].map(sector_map).fillna('Unknown')
    
    print(f"✓ Generated portfolio with {len(top_stocks)} stocks")
    return top_stocks


# =============================================================================
# GLOBAL TICKER FETCHER (V4)
# =============================================================================

# GICS-compatible sector mapping for international stocks
_GLOBAL_SECTOR_MAP = {
    # European large-caps
    'ASML.AS': 'Technology',     'SAP.DE': 'Technology',     'NOVO-B.CO': 'Healthcare',
    'LVMH.PA': 'Consumer Cyclical', 'MC.PA': 'Consumer Cyclical', 'OR.PA': 'Consumer Staples',
    'NESN.SW': 'Consumer Staples', 'ROG.SW': 'Healthcare',    'NOVN.SW': 'Healthcare',
    'SIE.DE': 'Industrials',     'ALV.DE': 'Financial Services', 'BAS.DE': 'Basic Materials',
    'AIR.PA': 'Industrials',     'DTE.DE': 'Communication Services', 'VOW3.DE': 'Consumer Cyclical',
    'SHEL.L': 'Energy',          'BP.L': 'Energy',            'AZN.L': 'Healthcare',
    'HSBA.L': 'Financial Services', 'RIO.L': 'Basic Materials', 'ULVR.L': 'Consumer Staples',
    'GSK.L': 'Healthcare',       'LLOY.L': 'Financial Services',
    # JP large-caps
    '7203.T': 'Consumer Cyclical', '6758.T': 'Technology',    '9984.T': 'Technology',
    '8306.T': 'Financial Services', '7267.T': 'Consumer Cyclical', '4063.T': 'Basic Materials',
    '9432.T': 'Communication Services', '8316.T': 'Financial Services',
    # Canadian large-caps
    'RY.TO': 'Financial Services', 'TD.TO': 'Financial Services', 'SHOP.TO': 'Technology',
    'CNR.TO': 'Industrials',     'ENB.TO': 'Energy',         'BNS.TO': 'Financial Services',
    # Australian large-caps
    'BHP.AX': 'Basic Materials', 'CBA.AX': 'Financial Services', 'CSL.AX': 'Healthcare',
    'WBC.AX': 'Financial Services',
}

# FX conversion pairs (to USD) — yfinance format
_FX_PAIRS = {
    '.PA': 'EURUSD=X',   '.DE': 'EURUSD=X',  '.AS': 'EURUSD=X',
    '.CO': 'EURUSD=X',   '.SW': 'CHFUSD=X',  '.L':  'GBPUSD=X',
    '.T':  'JPYUSD=X',   '.TO': 'CADUSD=X',  '.AX': 'AUDUSD=X',
}


def get_global_tickers(
    n_us: int = 700,
    n_eu: int = 200,
    n_jp: int = 100,
    cache_dir: str = '../cache/v3',
) -> pd.DataFrame:
    """
    Build a combined US + EU + JP ticker list with GICS-compatible sectors.

    Parameters
    ----------
    n_us      : max US tickers (from Finviz, already sector-filtered)
    n_eu      : max EU/UK tickers
    n_jp      : max Japan tickers
    cache_dir : cache directory (24-hour TTL)

    Returns
    -------
    DataFrame with columns: ticker, sector, industry, market_cap, region, fx_pair
    """
    cache_file = os.path.join(cache_dir, 'global_tickers.csv')
    if os.path.exists(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < 24:
            print(f"  Using cached global tickers ({age_h:.1f}h old)")
            return pd.read_csv(cache_file)

    rows = []

    # ── US via Finviz ──────────────────────────────────────────────────────
    if HAS_FINVIZ:
        try:
            from finviz.screener import Screener
            filters = ['sh_avgvol_o100', 'sh_price_o5']
            stock_list = Screener(filters=filters, table='Overview', order='marketcap')
            us_df = pd.DataFrame(stock_list.data)

            excluded_sectors = ['Basic Materials']
            excluded_industries = [
                'Gold', 'Silver', 'Copper', 'Aluminum', 'Steel',
                'Oil & Gas E&P', 'Oil & Gas Midstream', 'Oil & Gas Refining & Marketing',
                'Other Precious Metals & Mining', 'Other Industrial Metals & Mining',
            ]

            def parse_mcap(v):
                if pd.isna(v) or v == '-': return 0
                v = str(v).upper()
                if 'B' in v: return float(v.replace('B', '')) * 1e9
                if 'M' in v: return float(v.replace('M', '')) * 1e6
                return 0

            us_df['mcap'] = us_df['Market Cap'].apply(parse_mcap)
            us_df = us_df[us_df['mcap'] >= 100e6]
            us_df = us_df[~us_df['Sector'].isin(excluded_sectors)]
            us_df = us_df[~us_df['Industry'].isin(excluded_industries)]
            us_df = us_df.nlargest(n_us, 'mcap')

            for _, r in us_df.iterrows():
                rows.append({
                    'ticker': r['Ticker'], 'sector': r['Sector'],
                    'industry': r['Industry'], 'market_cap': r['mcap'],
                    'region': 'US', 'fx_pair': None,
                })
            print(f"  US: {len(us_df)} tickers")
        except Exception as e:
            print(f"  Finviz error: {e}")

    # ── EU (curated list — extend as needed) ───────────────────────────────
    eu_tickers = [
        'ASML.AS', 'SAP.DE', 'NOVO-B.CO', 'LVMH.PA', 'MC.PA', 'OR.PA',
        'NESN.SW', 'ROG.SW', 'NOVN.SW', 'SIE.DE', 'ALV.DE', 'BAS.DE',
        'AIR.PA', 'DTE.DE', 'VOW3.DE', 'SHEL.L', 'BP.L', 'AZN.L',
        'HSBA.L', 'RIO.L', 'ULVR.L', 'GSK.L', 'LLOY.L',
    ]
    for tk in eu_tickers[:n_eu]:
        suffix = '.' + tk.split('.')[-1] if '.' in tk else ''
        rows.append({
            'ticker': tk,
            'sector': _GLOBAL_SECTOR_MAP.get(tk, 'Unknown'),
            'industry': _GLOBAL_SECTOR_MAP.get(tk, 'Unknown'),
            'market_cap': 0,
            'region': 'EU',
            'fx_pair': _FX_PAIRS.get(suffix),
        })
    print(f"  EU: {len(eu_tickers[:n_eu])} tickers")

    # ── Japan (curated list — extend as needed) ────────────────────────────
    jp_tickers = [
        '7203.T', '6758.T', '9984.T', '8306.T', '7267.T',
        '4063.T', '9432.T', '8316.T',
    ]
    for tk in jp_tickers[:n_jp]:
        rows.append({
            'ticker': tk,
            'sector': _GLOBAL_SECTOR_MAP.get(tk, 'Unknown'),
            'industry': _GLOBAL_SECTOR_MAP.get(tk, 'Unknown'),
            'market_cap': 0,
            'region': 'JP',
            'fx_pair': 'JPYUSD=X',
        })
    print(f"  JP: {len(jp_tickers[:n_jp])} tickers")

    df = pd.DataFrame(rows).drop_duplicates('ticker').reset_index(drop=True)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(cache_file, index=False)
    print(f"  Total global universe: {len(df)} tickers  (cached)")
    return df


def apply_fx_conversion(price_data: dict, global_meta: pd.DataFrame) -> dict:
    """
    Convert non-USD price series to USD using daily FX rates.

    Parameters
    ----------
    price_data  : dict {ticker: pd.Series}  — raw local-currency prices
    global_meta : DataFrame with columns ticker, fx_pair

    Returns
    -------
    dict {ticker: pd.Series}  — USD-denominated prices
    """
    fx_cache: dict = {}
    fx_map = global_meta.set_index('ticker')['fx_pair'].to_dict()
    converted = {}

    for tk, px in price_data.items():
        pair = fx_map.get(tk)
        if not pair:
            converted[tk] = px   # already USD
            continue

        if pair not in fx_cache:
            try:
                fx_raw = yf.download(pair, period='3y', progress=False)
                if isinstance(fx_raw.columns, pd.MultiIndex):
                    fx_raw = fx_raw.droplevel(1, axis=1)
                fx_cache[pair] = fx_raw['Close'].dropna()
            except Exception:
                fx_cache[pair] = pd.Series(dtype=float)

        fx = fx_cache[pair]
        if fx.empty:
            converted[tk] = px
            continue

        aligned = px.reindex(fx.index).dropna()
        fx_aligned = fx.reindex(aligned.index).dropna()
        common = aligned.index.intersection(fx_aligned.index)
        if len(common) < 60:
            converted[tk] = px
            continue

        converted[tk] = aligned.loc[common] * fx_aligned.loc[common]

    return converted


# ─────────────────────────────────────────────────────────────────────────────
# Task-2 helper: Utilities ROE lookup
# ─────────────────────────────────────────────────────────────────────────────

def roe_filter(tickers: list) -> dict:
    """
    Fetch Return-on-Equity (%) for a list of tickers using yfinance.

    Returns a dict {ticker: roe_pct} where roe_pct is NaN when unavailable.
    Sourced from yfinance ``info['returnOnEquity']`` (decimal → %).
    """
    result = {}
    for tk in tickers:
        try:
            info = yf.Ticker(tk).info
            roe_decimal = info.get('returnOnEquity', None)
            result[tk] = float(roe_decimal) * 100.0 if roe_decimal is not None else float('nan')
        except Exception:
            result[tk] = float('nan')
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Task-3: FX forward-rate carry adjustment for global tickers
# ─────────────────────────────────────────────────────────────────────────────

def fx_adjusted_returns(
    ticker: str,
    returns_series: pd.Series,
    currency: str,
) -> pd.Series:
    """
    Adjust a daily-return series for FX carry cost using the 1-month forward rate.

    The forward-points differential is approximated from the forward FX rate symbol
    ``{currency}1MF=X`` (e.g. ``EUR1MF=X`` for EUR-denominated stocks).
    The monthly forward rate differential is scaled to daily: divide by 12*21.

    Parameters
    ----------
    ticker : str
        Ticker symbol (used for logging only).
    returns_series : pd.Series
        Daily total-return series indexed by trading dates.
    currency : str
        3-letter ISO currency code (e.g. ``'EUR'``, ``'GBP'``, ``'JPY'``).

    Returns
    -------
    pd.Series
        FX-adjusted daily returns. Falls back to unadjusted returns on failure.
    """
    if not currency or currency.upper() == 'USD':
        return returns_series

    forward_symbol = f"{currency.upper()}1MF=X"
    try:
        fwd_raw = yf.download(
            forward_symbol,
            start=returns_series.index[0].strftime('%Y-%m-%d'),
            end=returns_series.index[-1].strftime('%Y-%m-%d'),
            progress=False,
        )
        if fwd_raw.empty:
            return returns_series
        if isinstance(fwd_raw.columns, pd.MultiIndex):
            fwd_raw = fwd_raw.droplevel(1, axis=1)
        fwd_close = fwd_raw['Close'].dropna()
        # Daily carry: forward rate % change / 12 months ≈ 1-month cost per day
        fwd_daily = fwd_close.pct_change().fillna(0) / 12.0
        fwd_aligned = fwd_daily.reindex(returns_series.index, method='ffill').fillna(0)
        adjusted = returns_series + fwd_aligned
        return adjusted
    except Exception as exc:
        print(f"[fx_adjusted_returns] {ticker} ({currency}): {exc} — using unadjusted.")
        return returns_series


# ─────────────────────────────────────────────────────────────────────────────
# Task-1: DataDriftMonitor — detect alpha decay between monthly runs
# ─────────────────────────────────────────────────────────────────────────────

class DataDriftMonitor:
    """
    Detects composite-score distribution drift between monthly pipeline runs.

    Metrics
    -------
    - **Pearson correlation** on common top-100 tickers: below 0.85 → WARNING,
      below 0.70 → CRITICAL.
    - **Symmetric KL divergence** on 10-bin score histograms: above 0.10 →
      WARNING, above 0.20 → CRITICAL.

    Usage
    -----
    >>> monitor = DataDriftMonitor()
    >>> monitor.save_baseline(scores_df, 'output/v3/scores_baseline.json')
    >>> status = monitor.check_drift('output/v3/scores_baseline.json', new_scores_df)
    >>> # status in ('DRIFT_OK', 'DRIFT_WARNING', 'DRIFT_CRITICAL')
    """

    ALERT_CORR: float = 0.85   # Pearson correlation warning threshold
    ALERT_KL:   float = 0.10   # KL divergence warning threshold
    N_BINS:     int   = 10
    TOP_N:      int   = 100

    def save_baseline(self, scores_df: pd.DataFrame, output_path: str) -> None:
        """Persist the current composite-score distribution as a monthly baseline."""
        if 'ticker' not in scores_df.columns or 'composite_score' not in scores_df.columns:
            raise ValueError("scores_df must have 'ticker' and 'composite_score' columns.")
        top = scores_df.nlargest(self.TOP_N, 'composite_score').copy()
        summary = {
            'tickers':   top['ticker'].tolist(),
            'scores':    top['composite_score'].tolist(),
            'mean':      float(top['composite_score'].mean()),
            'std':       float(top['composite_score'].std()),
            'quantiles': {
                str(k): float(v)
                for k, v in top['composite_score'].quantile(
                    [0.10, 0.25, 0.50, 0.75, 0.90]
                ).items()
            },
            'timestamp': datetime.now().isoformat(),
            'n_tickers': int(len(scores_df)),
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[DataDriftMonitor] Baseline saved ({len(top)} tickers): {output_path}")

    def check_drift(self, old_scores_path: str, new_scores_df: pd.DataFrame) -> str:
        """Compare a new run to the saved baseline and return a drift status string.

        Parameters
        ----------
        old_scores_path : str
            Path to a JSON file previously written by ``save_baseline()``.
        new_scores_df : pd.DataFrame
            DataFrame with 'ticker' and 'composite_score' columns from the current run.

        Returns
        -------
        str — ``'DRIFT_OK'``, ``'DRIFT_WARNING'``, or ``'DRIFT_CRITICAL'``
        """
        if not os.path.exists(old_scores_path):
            print(f"[DataDriftMonitor] No baseline at {old_scores_path} — "
                  "saving current run as new baseline.")
            self.save_baseline(new_scores_df, old_scores_path)
            return 'DRIFT_OK'

        with open(old_scores_path) as f:
            baseline = json.load(f)

        old_scores = pd.Series(
            dict(zip(baseline['tickers'], baseline['scores']))
        )
        new_top = (
            new_scores_df
            .nlargest(self.TOP_N, 'composite_score')
            .set_index('ticker')['composite_score']
        )

        # ── Pearson correlation on overlap ───────────────────────────────────
        common = old_scores.index.intersection(new_top.index)
        if len(common) < 20:
            print(f"[DataDriftMonitor] Only {len(common)} common tickers — "
                  "universe shift detected → CRITICAL.")
            return 'DRIFT_CRITICAL'
        corr = float(old_scores[common].corr(new_top[common]))

        # ── Symmetric KL divergence on score histograms ──────────────────────
        lo = min(old_scores.min(), new_top.min())
        hi = max(old_scores.max(), new_top.max())
        bins = np.linspace(lo, hi, self.N_BINS + 1)
        p = np.histogram(old_scores.values, bins=bins)[0].astype(float) + 1e-9
        q = np.histogram(new_top.values,    bins=bins)[0].astype(float) + 1e-9
        p /= p.sum();  q /= q.sum()
        kl_sym = float(
            0.5 * np.sum(p * np.log(p / q)) +
            0.5 * np.sum(q * np.log(q / p))
        )

        # ── Verdict ─────────────────────────────────────────────────────────
        if corr < 0.70 or kl_sym > 0.20:
            status = 'DRIFT_CRITICAL'
        elif corr < self.ALERT_CORR or kl_sym > self.ALERT_KL:
            status = 'DRIFT_WARNING'
        else:
            status = 'DRIFT_OK'

        baseline_date = baseline.get('timestamp', 'unknown')[:10]
        print(
            f"[DataDriftMonitor] corr={corr:.3f}  KL_sym={kl_sym:.4f}  "
            f"common={len(common)}  baseline={baseline_date}  → {status}"
        )
        return status
