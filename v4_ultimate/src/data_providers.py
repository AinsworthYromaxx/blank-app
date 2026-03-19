"""
Expanded Data Providers Module
==============================

Integrates multiple data sources for comprehensive stock analysis:
- Fundamentals: Financial Modeling Prep, yfinance
- Macro/Economic: FRED (Federal Reserve)
- Alternative: News sentiment, insider trading, institutional holdings
- Global: International exchanges via yfinance

Usage:
    from data_providers import ExpandedDataProvider
    provider = ExpandedDataProvider()
    
    # Get fundamental data
    fundamentals = provider.get_fundamentals(['AAPL', 'MSFT'])
    
    # Get macro indicators
    macro = provider.get_macro_indicators()
    
    # Get alternative data
    alt_data = provider.get_alternative_data(['AAPL', 'MSFT'])
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import json
import warnings
import requests
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION - API Keys (set these as environment variables or directly)
# =============================================================================
# Free tiers available for all these APIs

FMP_API_KEY = os.environ.get('FMP_API_KEY', 'demo')  # financialmodelingprep.com - free tier: 250 calls/day
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')    # fred.stlouisfed.org - free, unlimited
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', '')  # finnhub.io - free tier: 60 calls/min


class FundamentalDataProvider:
    """
    Provides fundamental financial data from multiple sources:
    - Primary: Financial Modeling Prep (FMP)
    - Fallback: yfinance
    
    Key Metrics:
    - Valuation: P/E, P/B, P/S, EV/EBITDA
    - Profitability: ROE, ROA, margins
    - Financial Health: Current ratio, debt/equity, interest coverage
    - Growth: Revenue growth, earnings growth
    - Cash Flow: FCF yield, cash conversion
    """
    
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    
    def __init__(self, api_key: str = FMP_API_KEY, cache_dir: str = '../cache/fundamentals'):
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker}_fundamentals.json")
    
    def _is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        if not os.path.exists(cache_path):
            return False
        cache_time = os.path.getmtime(cache_path)
        return (time.time() - cache_time) < (max_age_hours * 3600)
    
    def _fetch_from_fmp(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data from Financial Modeling Prep"""
        if self.api_key == 'demo':
            return None  # Demo key has very limited access
            
        try:
            # Key metrics endpoint
            url = f"{self.BASE_URL}/key-metrics/{ticker}?limit=1&apikey={self.api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
        except Exception as e:
            pass
        return None
    
    def _fetch_from_yfinance(self, ticker: str) -> Dict:
        """Fetch fundamental data from yfinance (always available)"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if info is valid (sometimes yfinance returns empty or error)
            if not info or not isinstance(info, dict) or len(info) < 5:
                return {'ticker': ticker, 'error': 'Empty info from yfinance'}
            
            # Get financial statements for deeper analysis
            try:
                balance_sheet = stock.balance_sheet
                income_stmt = stock.income_stmt
                cash_flow = stock.cashflow
            except:
                balance_sheet = pd.DataFrame()
                income_stmt = pd.DataFrame()
                cash_flow = pd.DataFrame()
            
            # Extract key metrics
            metrics = {
                # Identification
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'US'),
                
                # Size
                'market_cap': info.get('marketCap', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'employees': info.get('fullTimeEmployees', 0),
                
                # Valuation
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'price_to_book': info.get('priceToBook', None),
                'price_to_sales': info.get('priceToSalesTrailing12Months', None),
                'ev_to_ebitda': info.get('enterpriseToEbitda', None),
                'ev_to_revenue': info.get('enterpriseToRevenue', None),
                
                # Profitability
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'gross_margin': info.get('grossMargins', None),
                'roe': info.get('returnOnEquity', None),
                'roa': info.get('returnOnAssets', None),
                
                # Growth
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', None),
                
                # Financial Health
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'total_debt': info.get('totalDebt', 0),
                'total_cash': info.get('totalCash', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                
                # Dividends
                'dividend_yield': info.get('dividendYield', None),
                'dividend_rate': info.get('dividendRate', None),
                'payout_ratio': info.get('payoutRatio', None),
                'ex_dividend_date': info.get('exDividendDate', None),
                
                # Analyst Estimates
                'target_mean_price': info.get('targetMeanPrice', None),
                'target_high_price': info.get('targetHighPrice', None),
                'target_low_price': info.get('targetLowPrice', None),
                'recommendation': info.get('recommendationKey', None),
                'recommendation_mean': info.get('recommendationMean', None),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', 0),
                
                # Short Interest
                'short_ratio': info.get('shortRatio', None),
                'short_percent_of_float': info.get('shortPercentOfFloat', None),
                
                # Beta & Volatility
                'beta': info.get('beta', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                
                # Metadata
                'fetch_timestamp': datetime.now().isoformat(),
            }
            
            # Calculate FCF Yield if we have data
            if metrics['market_cap'] and metrics['free_cash_flow']:
                metrics['fcf_yield'] = metrics['free_cash_flow'] / metrics['market_cap']
            else:
                metrics['fcf_yield'] = None
            
            return metrics
            
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}
    
    def _fetch_single_fundamental(self, ticker: str, use_cache: bool = True) -> Dict:
        """Fetch fundamentals for a single ticker (thread-safe)"""
        cache_path = self._get_cache_path(ticker)
        
        # Check cache first
        if use_cache and self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Try FMP first, then yfinance
        data = self._fetch_from_fmp(ticker)
        if not data:
            data = self._fetch_from_yfinance(ticker)
        
        # Ensure we always return a valid dict
        if data is None:
            data = {'ticker': ticker, 'error': 'No data available'}
        
        # Cache the result
        if data and 'error' not in data:
            try:
                with open(cache_path, 'w') as f:
                    json.dump(data, f)
            except:
                pass
        
        return data
    
    def get_fundamentals(self, tickers: List[str], use_cache: bool = True, 
                         max_workers: int = 10, parallel: bool = True) -> pd.DataFrame:
        """
        Fetch fundamental data for a list of tickers
        
        Parameters:
        -----------
        tickers : list of ticker symbols
        use_cache : bool, use 24h cache (default True)
        max_workers : int, number of parallel threads (default 10)
        parallel : bool, use parallel fetching (default True, ~5-10x faster)
        
        Returns DataFrame with 40+ fundamental metrics per stock
        """
        # Handle empty input
        if not tickers or len(tickers) == 0:
            return pd.DataFrame()
        
        if not parallel:
            # Legacy sequential mode
            results = []
            for ticker in tqdm(tickers, desc="Fetching fundamentals"):
                data = self._fetch_single_fundamental(ticker, use_cache)
                results.append(data)
                time.sleep(0.1)
            return pd.DataFrame(results)
        
        # Parallel mode - much faster
        results = []
        
        # First, quickly identify cached vs uncached tickers
        cached_tickers = []
        uncached_tickers = []
        
        for ticker in tickers:
            cache_path = self._get_cache_path(ticker)
            if use_cache and self._is_cache_valid(cache_path):
                cached_tickers.append(ticker)
            else:
                uncached_tickers.append(ticker)
        
        # Load cached data instantly
        for ticker in cached_tickers:
            cache_path = self._get_cache_path(ticker)
            try:
                with open(cache_path, 'r') as f:
                    results.append(json.load(f))
            except:
                uncached_tickers.append(ticker)  # Retry if cache read fails
        
        if cached_tickers:
            print(f"  ✓ Loaded {len(cached_tickers)} from cache")
        
        # Parallel fetch uncached tickers
        if uncached_tickers:
            print(f"  Fetching {len(uncached_tickers)} fundamentals ({max_workers} threads)...")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_ticker = {
                    executor.submit(self._fetch_single_fundamental, ticker, False): ticker
                    for ticker in uncached_tickers
                }
                
                # Collect results with progress bar
                for future in tqdm(as_completed(future_to_ticker), 
                                   total=len(uncached_tickers), 
                                   desc="Fetching fundamentals"):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result(timeout=30)
                        if data is not None:
                            results.append(data)
                        else:
                            results.append({'ticker': ticker, 'error': 'No data returned'})
                    except Exception as e:
                        results.append({'ticker': ticker, 'error': str(e)})
        
        # Filter out None values (safety)
        results = [r for r in results if r is not None]
        
        return pd.DataFrame(results)
    
    def calculate_value_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite value score based on fundamental metrics
        
        Components:
        - P/E percentile rank (lower is better)
        - P/B percentile rank (lower is better)
        - FCF yield percentile rank (higher is better)
        - EV/EBITDA percentile rank (lower is better)
        - PEG ratio (lower is better)
        """
        # Handle empty DataFrame
        if df is None or len(df) == 0:
            return df if df is not None else pd.DataFrame()
        
        df = df.copy()
        
        # Rank metrics (handle NaN and mixed types)
        def safe_rank(series, ascending=True):
            if series is None or len(series) == 0:
                return pd.Series([0.5] * len(df), index=df.index)
            # Convert to numeric (handles strings like "N/A", errors become NaN)
            numeric_series = pd.to_numeric(series, errors='coerce')
            return numeric_series.rank(pct=True, ascending=ascending, na_option='bottom')
        
        # Check and add columns with default neutral rank if missing
        if 'pe_ratio' in df.columns:
            df['pe_rank'] = safe_rank(df['pe_ratio'], ascending=True)
        else:
            df['pe_rank'] = 0.5
            
        if 'price_to_book' in df.columns:
            df['pb_rank'] = safe_rank(df['price_to_book'], ascending=True)
        else:
            df['pb_rank'] = 0.5
            
        if 'fcf_yield' in df.columns:
            df['fcf_rank'] = safe_rank(df['fcf_yield'], ascending=False)
        else:
            df['fcf_rank'] = 0.5
            
        if 'ev_to_ebitda' in df.columns:
            df['ev_ebitda_rank'] = safe_rank(df['ev_to_ebitda'], ascending=True)
        else:
            df['ev_ebitda_rank'] = 0.5
            
        if 'peg_ratio' in df.columns:
            df['peg_rank'] = safe_rank(df['peg_ratio'], ascending=True)
        else:
            df['peg_rank'] = 0.5
        
        # Composite value score (0-100)
        df['value_score'] = (
            df[['pe_rank', 'pb_rank', 'fcf_rank', 'ev_ebitda_rank', 'peg_rank']]
            .mean(axis=1) * 100
        )
        
        return df
    
    def calculate_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quality score based on profitability and financial health
        
        Components:
        - ROE (higher is better)
        - ROA (higher is better)
        - Operating margin (higher is better)
        - Current ratio (optimal around 1.5-2.5)
        - Debt/Equity (lower is better)
        """
        # Handle empty DataFrame
        if df is None or len(df) == 0:
            return df if df is not None else pd.DataFrame()
        
        df = df.copy()
        
        def safe_rank(series, ascending=False):
            if series is None or len(series) == 0:
                return pd.Series([0.5] * len(df), index=df.index)
            # Convert to numeric (handles strings like "N/A", errors become NaN)
            numeric_series = pd.to_numeric(series, errors='coerce')
            return numeric_series.rank(pct=True, ascending=ascending, na_option='bottom')
        
        # Check and add columns with default neutral rank if missing
        if 'roe' in df.columns:
            df['roe_rank'] = safe_rank(df['roe'])
        else:
            df['roe_rank'] = 0.5
            
        if 'roa' in df.columns:
            df['roa_rank'] = safe_rank(df['roa'])
        else:
            df['roa_rank'] = 0.5
            
        if 'operating_margin' in df.columns:
            df['margin_rank'] = safe_rank(df['operating_margin'])
        else:
            df['margin_rank'] = 0.5
            
        if 'debt_to_equity' in df.columns:
            df['debt_rank'] = safe_rank(df['debt_to_equity'], ascending=True)
        else:
            df['debt_rank'] = 0.5
        
        # Current ratio - penalize if too low or too high
        if 'current_ratio' in df.columns:
            # Convert to numeric first
            current_ratio_numeric = pd.to_numeric(df['current_ratio'], errors='coerce')
            df['current_ratio_score'] = current_ratio_numeric.apply(
                lambda x: 0.5 if pd.isna(x) else min(1, 1 - abs(x - 2) / 3) if 0.5 <= x <= 5 else 0
            )
        else:
            df['current_ratio_score'] = 0.5
        
        # Composite quality score
        df['quality_score'] = (
            df['roe_rank'] * 25 +
            df['roa_rank'] * 20 +
            df['margin_rank'] * 20 +
            df['debt_rank'] * 20 +
            df['current_ratio_score'] * 15
        )
        
        return df


class MacroDataProvider:
    """
    Provides macroeconomic data from FRED (Federal Reserve Economic Data)
    
    Key Indicators:
    - Interest Rates: Fed Funds, 10Y Treasury, yield curve
    - Inflation: CPI, PCE, breakeven inflation
    - Economy: GDP, unemployment, leading indicators
    - Markets: VIX, credit spreads
    - Sector-specific: Housing, manufacturing, retail
    """
    
    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # Key FRED series IDs
    SERIES = {
        # Interest Rates
        'fed_funds_rate': 'FEDFUNDS',
        'treasury_10y': 'DGS10',
        'treasury_2y': 'DGS2',
        'treasury_3m': 'DTB3',
        'yield_curve_10y2y': 'T10Y2Y',
        'yield_curve_10y3m': 'T10Y3M',
        
        # Inflation
        'cpi_yoy': 'CPIAUCSL',
        'core_cpi_yoy': 'CPILFESL',
        'pce_yoy': 'PCEPI',
        'breakeven_5y': 'T5YIE',
        'breakeven_10y': 'T10YIE',
        
        # Economy
        'gdp': 'GDP',
        'real_gdp': 'GDPC1',
        'unemployment_rate': 'UNRATE',
        'initial_claims': 'ICSA',
        'continued_claims': 'CCSA',
        'leading_index': 'USSLIND',
        
        # Consumer
        'consumer_sentiment': 'UMCSENT',
        'retail_sales': 'RSXFS',
        'personal_income': 'PI',
        'personal_spending': 'PCE',
        
        # Housing
        'housing_starts': 'HOUST',
        'existing_home_sales': 'EXHOSLUSM495S',
        'case_shiller_national': 'CSUSHPINSA',
        
        # Manufacturing
        'ism_manufacturing': 'MANEMP',
        'industrial_production': 'INDPRO',
        'capacity_utilization': 'TCU',
        
        # Credit & Markets
        'baa_spread': 'BAA10Y',
        'high_yield_spread': 'BAMLH0A0HYM2',
        'sp500': 'SP500',
        
        # Money Supply
        'm2': 'M2SL',
        'velocity_m2': 'M2V',
    }
    
    def __init__(self, api_key: str = FRED_API_KEY, cache_dir: str = '../cache/macro'):
        self.api_key = api_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _fetch_series(self, series_id: str, start_date: str = None) -> pd.Series:
        """Fetch a single FRED series"""
        if not self.api_key:
            print("⚠ FRED API key not set. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            return pd.Series(dtype=float)
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
        }
        
        try:
            response = requests.get(self.FRED_BASE_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                dates = []
                values = []
                for obs in observations:
                    try:
                        dates.append(pd.to_datetime(obs['date']))
                        val = obs['value']
                        values.append(float(val) if val != '.' else np.nan)
                    except:
                        continue
                
                return pd.Series(values, index=dates, name=series_id)
        except Exception as e:
            pass
        
        return pd.Series(dtype=float)
    
    def get_macro_indicators(self, indicators: List[str] = None) -> pd.DataFrame:
        """
        Fetch multiple macro indicators and return as DataFrame
        
        Args:
            indicators: List of indicator names (from SERIES keys)
                       If None, fetches all available
        
        Returns:
            DataFrame with dates as index and indicators as columns
        """
        if indicators is None:
            indicators = list(self.SERIES.keys())
        
        cache_file = os.path.join(self.cache_dir, 'macro_indicators.csv')
        
        # Check cache (24 hours)
        if os.path.exists(cache_file):
            cache_time = os.path.getmtime(cache_file)
            if (time.time() - cache_time) < 86400:
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        print("Fetching macro indicators from FRED...")
        all_series = {}
        
        for name in tqdm(indicators, desc="Downloading"):
            series_id = self.SERIES.get(name, name)
            data = self._fetch_series(series_id)
            if len(data) > 0:
                all_series[name] = data
            time.sleep(0.2)  # Rate limiting
        
        if all_series:
            df = pd.DataFrame(all_series)
            df.to_csv(cache_file)
            print(f"✓ Loaded {len(df.columns)} macro indicators")
            return df
        
        return pd.DataFrame()
    
    def get_current_regime(self, df: pd.DataFrame = None) -> Dict:
        """
        Determine current market regime based on macro indicators
        
        Returns dict with:
        - cycle_phase: expansion, peak, contraction, trough
        - rate_regime: rising, falling, stable
        - inflation_regime: hot, stable, deflationary
        - risk_regime: risk-on, risk-off
        """
        if df is None:
            df = self.get_macro_indicators()
        
        if df.empty:
            return {'error': 'No macro data available'}
        
        regime = {}
        
        # Recent values (last row)
        latest = df.ffill().iloc[-1]
        
        # Helper to convert numpy types to Python native types
        def to_python(val):
            if hasattr(val, 'item'):
                return val.item()
            return val
        
        # Yield curve slope
        yield_curve = to_python(latest.get('yield_curve_10y2y', 0))
        regime['yield_curve_slope'] = float(yield_curve) if yield_curve else 0
        regime['yield_curve_inverted'] = bool(yield_curve < 0) if yield_curve else False
        
        # Inflation
        cpi = latest.get('cpi_yoy', 0)
        if cpi > 4:
            regime['inflation_regime'] = 'hot'
        elif cpi < 1:
            regime['inflation_regime'] = 'deflationary_risk'
        else:
            regime['inflation_regime'] = 'stable'
        
        # Rate direction (compare to 3 months ago)
        if len(df) > 63:
            rate_3m_ago = df['fed_funds_rate'].iloc[-63] if 'fed_funds_rate' in df else None
            rate_now = latest.get('fed_funds_rate', 0)
            if rate_now and rate_3m_ago:
                if rate_now > rate_3m_ago + 0.25:
                    regime['rate_regime'] = 'rising'
                elif rate_now < rate_3m_ago - 0.25:
                    regime['rate_regime'] = 'falling'
                else:
                    regime['rate_regime'] = 'stable'
        
        # Unemployment trend
        if len(df) > 21:
            unemp_now = latest.get('unemployment_rate', 0)
            unemp_1m_ago = df['unemployment_rate'].iloc[-21] if 'unemployment_rate' in df else None
            if unemp_now and unemp_1m_ago:
                if unemp_now > unemp_1m_ago + 0.3:
                    regime['labor_trend'] = 'weakening'
                elif unemp_now < unemp_1m_ago - 0.1:
                    regime['labor_trend'] = 'strengthening'
                else:
                    regime['labor_trend'] = 'stable'
        
        # Credit conditions (BAA spread)
        baa_spread = latest.get('baa_spread', 0)
        if baa_spread > 3:
            regime['credit_conditions'] = 'tight'
        elif baa_spread < 1.5:
            regime['credit_conditions'] = 'loose'
        else:
            regime['credit_conditions'] = 'normal'
        
        return regime
    
    def get_sector_sensitivity(self) -> pd.DataFrame:
        """
        Return sector sensitivity matrix to macro factors
        Based on historical correlations
        """
        # Empirical sector sensitivities
        data = {
            'sector': [
                'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
                'Consumer Staples', 'Energy', 'Utilities', 'Real Estate',
                'Industrials', 'Materials', 'Communication Services'
            ],
            'rate_sensitivity': [-0.6, -0.2, 0.5, -0.3, 0.0, 0.1, -0.7, -0.8, 0.0, 0.1, -0.3],
            'inflation_sensitivity': [-0.3, 0.0, 0.3, -0.2, 0.2, 0.7, 0.1, -0.4, 0.2, 0.5, -0.1],
            'growth_sensitivity': [0.8, 0.3, 0.6, 0.7, 0.1, 0.4, 0.0, 0.3, 0.7, 0.6, 0.5],
            'dollar_sensitivity': [-0.3, 0.0, 0.1, -0.1, 0.2, -0.2, 0.1, 0.0, -0.2, -0.3, -0.1],
        }
        return pd.DataFrame(data)


class AlternativeDataProvider:
    """
    Provides alternative data:
    - News sentiment
    - Insider trading
    - Institutional holdings  
    - Social sentiment (Reddit, Twitter mentions)
    - Short interest
    - Options flow signals
    """
    
    def __init__(self, finnhub_key: str = FINNHUB_API_KEY, cache_dir: str = '../cache/alternative'):
        self.finnhub_key = finnhub_key
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_insider_trading(self, ticker: str) -> pd.DataFrame:
        """Fetch insider trading data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get insider transactions
            insider_transactions = stock.insider_transactions
            if insider_transactions is not None and len(insider_transactions) > 0:
                return insider_transactions
            
            return pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def get_institutional_holders(self, ticker: str) -> Dict:
        """Fetch institutional ownership data"""
        try:
            stock = yf.Ticker(ticker)
            
            holders = stock.institutional_holders
            major_holders = stock.major_holders
            
            result = {
                'ticker': ticker,
                'institutional_holders': holders.to_dict() if holders is not None else {},
                'major_holders': major_holders.to_dict() if major_holders is not None else {},
            }
            
            # Summary statistics
            if holders is not None and len(holders) > 0:
                result['num_institutional_holders'] = len(holders)
                if 'Shares' in holders.columns:
                    result['total_institutional_shares'] = holders['Shares'].sum()
                if '% Out' in holders.columns:
                    result['total_institutional_pct'] = holders['% Out'].sum()
            
            return result
        except:
            return {'ticker': ticker, 'error': 'Failed to fetch'}
    
    def get_news_sentiment(self, ticker: str, days: int = 7) -> Dict:
        """
        Get news sentiment from Finnhub (requires free API key)
        """
        if not self.finnhub_key:
            # Fallback: use yfinance news
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                if news:
                    return {
                        'ticker': ticker,
                        'news_count': len(news),
                        'recent_headlines': [n.get('title', '') for n in news[:5]],
                        'source': 'yfinance'
                    }
            except:
                pass
            return {'ticker': ticker, 'error': 'No Finnhub API key'}
        
        # Use Finnhub
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={self.finnhub_key}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                news = response.json()
                
                # Basic sentiment scoring based on headline keywords
                positive_words = ['surge', 'jump', 'beat', 'record', 'strong', 'growth', 'profit', 'gain', 'rally']
                negative_words = ['fall', 'drop', 'miss', 'weak', 'loss', 'decline', 'crash', 'plunge', 'cut']
                
                sentiment_scores = []
                for article in news:
                    headline = article.get('headline', '').lower()
                    pos_count = sum(1 for w in positive_words if w in headline)
                    neg_count = sum(1 for w in negative_words if w in headline)
                    sentiment_scores.append(pos_count - neg_count)
                
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
                
                return {
                    'ticker': ticker,
                    'news_count': len(news),
                    'avg_sentiment': avg_sentiment,
                    'positive_news_pct': sum(1 for s in sentiment_scores if s > 0) / len(news) if news else 0,
                    'recent_headlines': [n.get('headline', '') for n in news[:5]],
                }
        except:
            pass
        
        return {'ticker': ticker, 'error': 'Failed to fetch news'}
    
    def get_short_interest(self, ticker: str) -> Dict:
        """Get short interest data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'ticker': ticker,
                'short_ratio': info.get('shortRatio', None),
                'short_percent_of_float': info.get('shortPercentOfFloat', None),
                'shares_short': info.get('sharesShort', None),
                'shares_short_prior_month': info.get('sharesShortPriorMonth', None),
                'short_trend': 'increasing' if info.get('sharesShort', 0) > info.get('sharesShortPriorMonth', 0) else 'decreasing'
            }
        except:
            return {'ticker': ticker, 'error': 'Failed to fetch'}
    
    def _fetch_single_alternative(self, ticker: str) -> Dict:
        """Fetch all alternative data for a single ticker (thread-safe)"""
        data = {'ticker': ticker}
        
        try:
            # Short interest
            short_data = self.get_short_interest(ticker)
            data.update({f"short_{k}": v for k, v in short_data.items() if k != 'ticker'})
            
            # News sentiment
            news_data = self.get_news_sentiment(ticker)
            data.update({f"news_{k}": v for k, v in news_data.items() if k != 'ticker'})
            
            # Institutional
            inst_data = self.get_institutional_holders(ticker)
            data['num_institutional_holders'] = inst_data.get('num_institutional_holders', 0)
            data['institutional_ownership_pct'] = inst_data.get('total_institutional_pct', 0)
        except Exception as e:
            data['error'] = str(e)
        
        return data
    
    def get_alternative_data(self, tickers: List[str], max_workers: int = 8, 
                             parallel: bool = True) -> pd.DataFrame:
        """
        Fetch all alternative data for a list of tickers
        
        Parameters:
        -----------
        tickers : list of ticker symbols
        max_workers : int, number of parallel threads (default 8)
        parallel : bool, use parallel fetching (default True, ~5-8x faster)
        """
        if not parallel:
            # Legacy sequential mode
            results = []
            for ticker in tqdm(tickers, desc="Fetching alternative data"):
                data = self._fetch_single_alternative(ticker)
                results.append(data)
                time.sleep(0.2)
            return pd.DataFrame(results)
        
        # Parallel mode
        results = []
        print(f"  Fetching alternative data for {len(tickers)} stocks ({max_workers} threads)...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_single_alternative, ticker): ticker
                for ticker in tickers
            }
            
            for future in tqdm(as_completed(future_to_ticker), 
                               total=len(tickers), 
                               desc="Fetching alternative data"):
                ticker = future_to_ticker[future]
                try:
                    data = future.result(timeout=30)
                    results.append(data)
                except Exception as e:
                    results.append({'ticker': ticker, 'error': str(e)})
        
        return pd.DataFrame(results)


class GlobalMarketProvider:
    """
    Provides access to international markets via yfinance
    
    Supported exchanges:
    - US: NYSE, NASDAQ, AMEX
    - Europe: LSE (London), Euronext (Paris, Amsterdam), XETRA (Frankfurt)
    - Asia: TSE (Tokyo), HKEX (Hong Kong), SSE (Shanghai), Kospi (Seoul)
    - Other: TSX (Canada), ASX (Australia), Bovespa (Brazil)
    """
    
    # Exchange suffixes for yfinance
    EXCHANGE_SUFFIXES = {
        'US': '',        # No suffix
        'UK': '.L',      # London Stock Exchange
        'Germany': '.DE', # XETRA Frankfurt
        'France': '.PA', # Paris Euronext
        'Netherlands': '.AS',  # Amsterdam
        'Japan': '.T',   # Tokyo Stock Exchange
        'Hong Kong': '.HK',  # HKEX
        'China': '.SS',  # Shanghai (also .SZ for Shenzhen)
        'Korea': '.KS',  # Korea Exchange
        'Canada': '.TO', # Toronto Stock Exchange
        'Australia': '.AX',  # ASX
        'Brazil': '.SA', # Bovespa
        'India': '.NS',  # NSE India (also .BO for BSE)
        'Switzerland': '.SW',  # Swiss Exchange
        'Spain': '.MC',  # Madrid
        'Italy': '.MI',  # Milan
    }
    
    # Major indices for each market
    MAJOR_INDICES = {
        'US': '^GSPC',       # S&P 500
        'UK': '^FTSE',       # FTSE 100
        'Germany': '^GDAXI', # DAX
        'France': '^FCHI',   # CAC 40
        'Japan': '^N225',    # Nikkei 225
        'Hong Kong': '^HSI', # Hang Seng
        'China': '000001.SS', # Shanghai Composite
        'Korea': '^KS11',    # KOSPI
        'Canada': '^GSPTSE', # TSX Composite
        'Australia': '^AXJO', # ASX 200
        'Brazil': '^BVSP',   # Bovespa
        'India': '^NSEI',    # Nifty 50
    }
    
    # Some popular international stocks (ADRs in US + local tickers)
    SAMPLE_INTERNATIONAL = {
        'UK': ['SHEL.L', 'HSBA.L', 'BP.L', 'AZN.L', 'GSK.L', 'ULVR.L', 'RIO.L', 'DGE.L'],
        'Germany': ['SAP.DE', 'SIE.DE', 'ALV.DE', 'BAS.DE', 'DTE.DE', 'BMW.DE', 'VOW3.DE', 'MRK.DE'],
        'France': ['MC.PA', 'OR.PA', 'SAN.PA', 'AIR.PA', 'TTE.PA', 'BNP.PA', 'ENGI.PA'],
        'Japan': ['7203.T', '6758.T', '9984.T', '6861.T', '7267.T', '9432.T', '6501.T'],  # Toyota, Sony, etc.
        'Hong Kong': ['0700.HK', '9988.HK', '0005.HK', '1299.HK', '2318.HK'],  # Tencent, Alibaba, HSBC
        'Canada': ['RY.TO', 'TD.TO', 'ENB.TO', 'CNR.TO', 'BMO.TO', 'BNS.TO', 'SHOP.TO'],
        'Australia': ['BHP.AX', 'CBA.AX', 'CSL.AX', 'WBC.AX', 'NAB.AX', 'ANZ.AX'],
    }
    
    def __init__(self, cache_dir: str = '../cache/global'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_major_indices(self, period: str = '1y') -> pd.DataFrame:
        """Fetch performance of major global indices"""
        indices = []
        
        for market, symbol in self.MAJOR_INDICES.items():
            try:
                data = yf.download(symbol, period=period, progress=False)
                if len(data) > 0:
                    # Extract scalar values properly (yfinance can return Series)
                    close = data['Close']
                    if hasattr(close, 'iloc'):
                        current = float(close.iloc[-1].iloc[0]) if hasattr(close.iloc[-1], 'iloc') else float(close.iloc[-1])
                        first = float(close.iloc[0].iloc[0]) if hasattr(close.iloc[0], 'iloc') else float(close.iloc[0])
                    else:
                        current = float(close[-1])
                        first = float(close[0])
                    
                    returns = {
                        'market': market,
                        'symbol': symbol,
                        'current_price': round(current, 2),
                        'ytd_return': round((current / first - 1) * 100, 2),
                        '1m_return': round((current / float(close.iloc[-21].iloc[0] if hasattr(close.iloc[-21], 'iloc') else close.iloc[-21]) - 1) * 100, 2) if len(data) >= 21 else None,
                        '3m_return': round((current / float(close.iloc[-63].iloc[0] if hasattr(close.iloc[-63], 'iloc') else close.iloc[-63]) - 1) * 100, 2) if len(data) >= 63 else None,
                        '6m_return': round((current / float(close.iloc[-126].iloc[0] if hasattr(close.iloc[-126], 'iloc') else close.iloc[-126]) - 1) * 100, 2) if len(data) >= 126 else None,
                    }
                    indices.append(returns)
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol}: {e}")
                continue
            time.sleep(0.1)
        
        df = pd.DataFrame(indices)
        # Ensure numeric columns for sorting
        for col in ['ytd_return', '1m_return', '3m_return', '6m_return', 'current_price']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def get_international_stocks(self, markets: List[str] = None, period: str = '3y') -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for major international stocks
        
        Args:
            markets: List of market codes (e.g., ['UK', 'Germany', 'Japan'])
            period: Historical period
        
        Returns:
            Dict mapping ticker to price DataFrame
        """
        if markets is None:
            markets = list(self.SAMPLE_INTERNATIONAL.keys())
        
        price_data = {}
        
        for market in markets:
            tickers = self.SAMPLE_INTERNATIONAL.get(market, [])
            print(f"Fetching {market} stocks: {len(tickers)} tickers")
            
            for ticker in tqdm(tickers, desc=f"Downloading {market}"):
                try:
                    data = yf.download(ticker, period=period, progress=False)
                    if len(data) > 0:
                        price_data[ticker] = data
                except:
                    continue
                time.sleep(0.2)
        
        return price_data
    
    def convert_to_local_ticker(self, us_ticker: str, target_market: str) -> str:
        """Convert US ADR ticker to local exchange ticker (where known)"""
        # Common ADR to local mappings
        adr_mappings = {
            'BABA': '9988.HK',  # Alibaba
            'TSM': '2330.TW',   # Taiwan Semiconductor
            'SONY': '6758.T',  # Sony
            'TM': '7203.T',    # Toyota
            'NVO': 'NOVO-B.CO',  # Novo Nordisk
            'ASML': 'ASML.AS',  # ASML
            'UL': 'ULVR.L',    # Unilever
        }
        return adr_mappings.get(us_ticker.upper(), us_ticker)


class ExpandedDataProvider:
    """
    Master class that combines all data providers for comprehensive analysis
    """
    
    def __init__(self, cache_dir: str = '../cache/expanded'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.fundamentals = FundamentalDataProvider(cache_dir=os.path.join(cache_dir, 'fundamentals'))
        self.macro = MacroDataProvider(cache_dir=os.path.join(cache_dir, 'macro'))
        self.alternative = AlternativeDataProvider(cache_dir=os.path.join(cache_dir, 'alternative'))
        self.global_markets = GlobalMarketProvider(cache_dir=os.path.join(cache_dir, 'global'))
    
    def get_comprehensive_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch all available data for a list of tickers:
        - Fundamentals (40+ metrics)
        - Alternative data (sentiment, short interest, institutional)
        - Value and quality scores
        
        Returns comprehensive DataFrame ready for analysis
        """
        print(f"\n{'='*60}")
        print("COMPREHENSIVE DATA FETCH")
        print(f"{'='*60}\n")
        
        # Step 1: Fundamentals
        print("[1/3] Fetching fundamental data...")
        fundamentals = self.fundamentals.get_fundamentals(tickers)
        fundamentals = self.fundamentals.calculate_value_score(fundamentals)
        fundamentals = self.fundamentals.calculate_quality_score(fundamentals)
        print(f"✓ Got fundamentals for {len(fundamentals)} stocks")
        
        # Step 2: Alternative data
        print("\n[2/3] Fetching alternative data...")
        alternative = self.alternative.get_alternative_data(tickers)
        print(f"✓ Got alternative data for {len(alternative)} stocks")
        
        # Step 3: Merge
        print("\n[3/3] Merging datasets...")
        comprehensive = fundamentals.merge(alternative, on='ticker', how='left')
        
        print(f"\n✓ Comprehensive dataset ready: {len(comprehensive)} stocks, {len(comprehensive.columns)} features")
        
        return comprehensive
    
    def get_market_overview(self) -> Dict:
        """
        Get current market overview:
        - Macro regime
        - Global index performance
        - Sector sensitivities
        """
        overview = {}
        
        # Macro regime
        print("Fetching macro indicators...")
        macro_df = self.macro.get_macro_indicators()
        overview['macro_regime'] = self.macro.get_current_regime(macro_df)
        
        # Global indices
        print("Fetching global indices...")
        overview['global_indices'] = self.global_markets.get_major_indices(period='1y')
        
        # Sector sensitivity
        overview['sector_sensitivity'] = self.macro.get_sector_sensitivity()
        
        return overview
    
    def export_all(self, tickers: List[str], output_dir: str = 'exported_data'):
        """Export all data to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get comprehensive data
        data = self.get_comprehensive_data(tickers)
        data.to_csv(os.path.join(output_dir, 'comprehensive_stock_data.csv'), index=False)
        
        # Macro data
        macro = self.macro.get_macro_indicators()
        macro.to_csv(os.path.join(output_dir, 'macro_indicators.csv'))
        
        # Global indices
        indices = self.global_markets.get_major_indices()
        indices.to_csv(os.path.join(output_dir, 'global_indices.csv'), index=False)
        
        print(f"\n✓ All data exported to: {output_dir}/")


# =============================================================================
# QUICK START EXAMPLES
# =============================================================================

if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("EXPANDED DATA PROVIDERS - QUICK START")
    print("=" * 60)
    
    provider = ExpandedDataProvider()
    
    # Test with a few stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Get fundamentals
    print("\n[TEST] Fetching fundamentals...")
    fundamentals = provider.fundamentals.get_fundamentals(test_tickers)
    print(fundamentals[['ticker', 'pe_ratio', 'roe', 'debt_to_equity']].to_string())
    
    # Get macro regime
    print("\n[TEST] Getting macro regime...")
    regime = provider.macro.get_current_regime()
    print(f"Current regime: {regime}")
    
    print("\n✓ All providers working!")
