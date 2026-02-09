# src/backtest/buy_today_sell_tomorrow_backtest.py
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BuyTodaySellTomorrowBacktester:
    def __init__(self, initial_capital=100000, max_positions=5, position_size_pct=0.1):
        """
        Initialize the BuyTodaySellTomorrow backtester
        
        Args:
            initial_capital: Starting capital (₹100,000 default)
            max_positions: Maximum number of simultaneous positions (5 default)
            position_size_pct: Percentage of capital per position (10% default)
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.transaction_cost = 0.001  # 0.1% per transaction
        self.reset()
    
    def reset(self):
        """Reset all backtest variables"""
        self.cash = self.initial_capital
        self.positions = []  # List of active positions
        self.trade_log = []  # History of all trades
        self.portfolio_values = []  # Portfolio value over time
        self.daily_performance = []  # Daily performance tracking
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for BuyTodaySellTomorrow strategy
        """
        df = df.copy()
        
        # Flatten multi-level columns if they exist (from yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            # Take the first level of the tuple (Close, High, Low, etc.)
            df.columns = df.columns.get_level_values(0)
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        
        # RSI (Relative Strength Index)
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self._calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'].div(df['Volume_MA']).fillna(1.0)
        
        # Volatility (ATR)
        df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'])
        
        # Price momentum
        df['Momentum'] = df['Close'].pct_change(periods=3)
        df['ROC'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        
        # Price position relative to moving averages
        df['Price_vs_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['BB_Position'] = (df['Close'] - df['BB_Middle']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Support and resistance levels
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        
        return df
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD and Signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        middle_band = sma
        lower_band = sma - (std * num_std)
        return upper_band, middle_band, lower_band
    
    def _calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate BuyTodaySellTomorrow signals based on technical indicators
        Simplified approach focusing on short-term reversals and momentum
        """
        df = df.copy()
        
        # Fill NaN values to avoid masking errors
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Buy signals - look for potential short-term upward moves
        # Based on: oversold RSI + positive momentum + volume confirmation
        df['Buy_Signal'] = (
            (df['RSI'] < 60) &           # Not extremely overbought
            (df['Momentum'] > -0.005) &  # Slight positive momentum
            (df['Volume_Ratio'] > 0.8) & # Normal or above volume
            (df['Close'] < df['SMA_5']) & # Price above short-term support
            (df['MACD'] > df['MACD_Signal'] - 0.1)  # Slight bullish MACD
        )
        
        # Sell signals - look for potential short-term downward moves
        # Based on: overbought RSI + negative momentum
        df['Sell_Signal'] = (
            (df['RSI'] > 40) &           # Not extremely oversold
            (df['Momentum'] < 0.005) &   # Slight negative momentum
            (df['Volume_Ratio'] > 0.8) & # Normal or above volume
            (df['Close'] > df['SMA_5']) & # Price below short-term resistance
            (df['MACD'] < df['MACD_Signal'] + 0.1)  # Slight bearish MACD
        )
        
        # Shift signals to represent next-day execution and fill NaN
        df['Buy_Signal_Next'] = df['Buy_Signal'].shift(1).fillna(False)
        df['Sell_Signal_Next'] = df['Sell_Signal'].shift(1).fillna(False)
        
        return df
    
    def calculate_confidence_score(self, row: pd.Series) -> float:
        """
        Calculate confidence score for a potential trade based on multiple factors
        """
        score = 0
        
        # RSI contribution (higher weight for extreme values)
        if row['RSI'] < 30:  # Oversold
            score += 25
        elif row['RSI'] < 40:  # Somewhat oversold
            score += 15
        elif row['RSI'] > 70:  # Overbought
            score += 25
        elif row['RSI'] > 60:  # Somewhat overbought
            score += 15
        
        # MACD contribution
        if row['MACD'] > row['MACD_Signal']:  # Bullish
            score += 15
        elif row['MACD'] < row['MACD_Signal']:  # Bearish
            score += 15
        
        # Bollinger Band position
        if abs(row['BB_Position']) > 0.8:  # At extremes
            score += 20
        elif abs(row['BB_Position']) > 0.5:  # Near extremes
            score += 10
        
        # Volume confirmation
        if row['Volume_Ratio'] > 1.5:  # High volume
            score += 15
        elif row['Volume_Ratio'] > 1.2:  # Above average
            score += 10
        
        # Momentum
        if abs(row['Momentum']) > 0.03:  # Strong momentum
            score += 10
        elif abs(row['Momentum']) > 0.01:  # Moderate momentum
            score += 5
        
        # Normalize to 0-100 scale
        return min(score, 100)
    
    def should_enter_position(self, df: pd.DataFrame, current_date: pd.Timestamp) -> Tuple[bool, Dict]:
        """
        Determine if we should enter a position based on signals and constraints
        """
        if len(self.positions) >= self.max_positions:
            return False, {}
        
        current_row = df[df.index == current_date]
        if current_row.empty:
            return False, {}
        
        current_row = current_row.iloc[0]
        
        # Check if buy signal exists and confidence is high enough
        if current_row['Buy_Signal_Next'] and self.calculate_confidence_score(current_row) >= 60:
            # Calculate position size based on available cash and risk management
            available_cash = self.cash
            position_size = min(available_cash * self.position_size_pct, available_cash)
            
            # Calculate number of shares to buy
            price = current_row['Close']
            shares = int(position_size // price)
            
            if shares > 0:
                entry_info = {
                    'entry_date': current_date.date(),
                    'entry_price': price,
                    'shares': shares,
                    'confidence_score': self.calculate_confidence_score(current_row),
                    'stop_loss': price * 0.97,  # 3% stop loss
                    'target_price': price * 1.03  # 3% target
                }
                return True, entry_info
        
        return False, {}
    
    def should_exit_position(self, position: Dict, current_date: pd.Timestamp, current_price: float) -> bool:
        """
        Determine if we should exit a position
        """
        # Exit if stop loss is hit
        if current_price <= position['stop_loss']:
            return True
        
        # Exit if target is reached
        if current_price >= position['target_price']:
            return True
        
        # Exit after holding for more than 2 days (since it's a buy-today-sell-tomorrow strategy)
        days_held = (current_date.date() - position['entry_date']).days
        if days_held >= 2:
            return True
        
        return False
    
    def backtest_stock(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest the BuyTodaySellTomorrow strategy for a single stock
        """
        self.reset()
        
        # Fetch historical data
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        
        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Calculate technical indicators
        stock_data = self.calculate_technical_indicators(stock_data)
        
        # Generate signals
        stock_data = self.generate_signals(stock_data)
        
        # Iterate through each trading day
        for date, row in stock_data.iterrows():
            current_price = row['Close']
            
            # Check for exits first (sell positions)
            positions_to_remove = []
            for i, position in enumerate(self.positions):
                if self.should_exit_position(position, date, current_price):
                    # Calculate profit/loss
                    pnl = (current_price - position['entry_price']) * position['shares']
                    transaction_cost = (position['entry_price'] + current_price) * self.transaction_cost
                    net_pnl = pnl - transaction_cost
                    
                    # Update cash
                    self.cash += (position['entry_price'] * position['shares']) + net_pnl
                    
                    # Log the trade
                    self.trade_log.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': date.date(),
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'pnl': net_pnl,
                        'return_pct': (net_pnl / (position['entry_price'] * position['shares'])) * 100,
                        'confidence_score': position['confidence_score']
                    })
                    
                    positions_to_remove.append(i)
            
            # Remove exited positions
            for i in sorted(positions_to_remove, reverse=True):
                del self.positions[i]
            
            # Check for new entries (buy positions)
            should_enter, entry_info = self.should_enter_position(stock_data, date)
            if should_enter:
                # Deduct cost from cash
                cost = entry_info['entry_price'] * entry_info['shares']
                transaction_cost = cost * self.transaction_cost
                total_cost = cost + transaction_cost
                
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.positions.append(entry_info)
        
        # Calculate final portfolio value
        total_shares_value = sum(pos['shares'] * current_price for pos in self.positions)
        final_portfolio_value = self.cash + total_shares_value
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(symbol, stock_data.index[0], stock_data.index[-1])
        
        return {
            'symbol': symbol,
            'performance': performance_metrics,
            'trade_log': self.trade_log,
            'final_portfolio_value': final_portfolio_value,
            'total_return': final_portfolio_value - self.initial_capital
        }
    
    def _calculate_performance_metrics(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
        """
        Calculate performance metrics for the backtest
        """
        if not self.trade_log:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'profit_factor': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_holding_period': 0
            }
        
        # Convert trade log to DataFrame for easier calculations
        trades_df = pd.DataFrame(self.trade_log)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Calculate returns
        total_return = trades_df['pnl'].sum()
        total_return_pct = (total_return / self.initial_capital) * 100
        avg_return_per_trade = trades_df['pnl'].mean()
        
        # Best and worst trades
        best_trade = trades_df['return_pct'].max() if not trades_df.empty else 0
        worst_trade = trades_df['return_pct'].min() if not trades_df.empty else 0
        
        # Average holding period
        avg_holding_period = trades_df.apply(
            lambda x: (pd.to_datetime(x['exit_date']) - pd.to_datetime(x['entry_date'])).days,
            axis=1
        ).mean() if not trades_df.empty else 0
        
        # Max drawdown calculation would require daily portfolio values
        # For simplicity, we'll use the largest single trade loss as a proxy
        max_single_loss = abs(trades_df['pnl'].min()) if not trades_df.empty else 0
        max_drawdown = (max_single_loss / self.initial_capital) * 100
        
        # Sharpe ratio calculation (simplified)
        # Assuming 6% annual risk-free rate
        if not trades_df.empty:
            daily_returns = trades_df['pnl'] / self.initial_capital
            excess_returns = daily_returns - (0.06 / 252)  # Daily risk-free rate
            if daily_returns.std() != 0:
                sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Sortino ratio calculation (uses downside deviation)
        if not trades_df.empty:
            daily_returns = trades_df['pnl'] / self.initial_capital
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std()
                if downside_deviation != 0:
                    excess_returns = daily_returns - (0.06 / 252)
                    sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)
                else:
                    sortino_ratio = sharpe_ratio  # If no negative returns, use sharpe ratio
            else:
                sortino_ratio = sharpe_ratio  # If no negative returns, use sharpe ratio
        else:
            sortino_ratio = 0
        
        # Profit factor calculation
        gross_profits = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if not trades_df.empty else 0
        gross_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if not trades_df.empty else 0
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return {
            'total_return': round(total_return, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'avg_return_per_trade': round(avg_return_per_trade, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'profit_factor': round(profit_factor, 2) if isinstance(profit_factor, (int, float)) and not np.isinf(profit_factor) else 'Inf',
            'best_trade': round(best_trade, 2),
            'worst_trade': round(worst_trade, 2),
            'avg_holding_period': round(avg_holding_period, 2)
        }
    
    def backtest_universe(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Backtest the strategy across multiple stocks
        """
        results = []
        
        print(f"Backtesting BuyTodaySellTomorrow strategy on {len(symbols)} stocks...")
        print(f"Period: {start_date} to {end_date}")
        print("="*80)
        
        for i, symbol in enumerate(symbols):
            print(f"Processing {symbol} ({i+1}/{len(symbols)})...")
            
            try:
                result = self.backtest_stock(symbol, start_date, end_date)
                
                if 'error' not in result:
                    perf = result['performance']
                    results.append({
                        'symbol': symbol,
                        'total_return': perf['total_return'],
                        'total_return_pct': perf['total_return_pct'],
                        'total_trades': perf['total_trades'],
                        'win_rate': perf['win_rate'],
                        'sharpe_ratio': perf['sharpe_ratio'],
                        'max_drawdown': perf['max_drawdown'],
                        'profit_factor': perf['profit_factor'],
                        'best_trade': perf['best_trade'],
                        'worst_trade': perf['worst_trade']
                    })
                    
                    print(f"  ✓ {symbol}: {perf['total_return_pct']:.2f}% return, "
                          f"{perf['win_rate']:.1f}% win rate, {perf['sharpe_ratio']:.2f} SR")
                else:
                    print(f"  ✗ {symbol}: {result['error']}")
                    
            except Exception as e:
                print(f"  ✗ {symbol}: Error - {str(e)}")
        
        # Calculate aggregate performance metrics
        if results:
            results_df = pd.DataFrame(results)
            
            print("\n" + "="*80)
            print("AGGREGATE PERFORMANCE METRICS")
            print("="*80)
            print(f"Total stocks tested: {len(results_df)}")
            print(f"Average return: {results_df['total_return_pct'].mean():.2f}%")
            print(f"Median return: {results_df['total_return_pct'].median():.2f}%")
            print(f"Total return across all stocks: {results_df['total_return'].sum():,.2f}")
            print(f"Avg win rate: {results_df['win_rate'].mean():.2f}%")
            print(f"Avg Sharpe ratio: {results_df['sharpe_ratio'].mean():.2f}")
            print(f"Avg max drawdown: {results_df['max_drawdown'].mean():.2f}%")
            
            positive_performers = len(results_df[results_df['total_return_pct'] > 0])
            print(f"Positive performers: {positive_performers}/{len(results_df)} "
                  f"({positive_performers/len(results_df)*100:.1f}%)")
            
            print(f"Best performer: {results_df.loc[results_df['total_return_pct'].idxmax(), 'symbol']} "
                  f"({results_df['total_return_pct'].max():.2f}%)")
            print(f"Worst performer: {results_df.loc[results_df['total_return_pct'].idxmin(), 'symbol']} "
                  f"({results_df['total_return_pct'].min():.2f}%)")
            
            # Strategy statistics
            total_trades = results_df['total_trades'].sum()
            avg_total_trades = results_df['total_trades'].mean()
            print(f"Total trades across all stocks: {total_trades}")
            print(f"Avg trades per stock: {avg_total_trades:.1f}")
            
            return {
                'individual_results': results,
                'aggregate_metrics': {
                    'total_stocks': len(results_df),
                    'avg_return': results_df['total_return_pct'].mean(),
                    'median_return': results_df['total_return_pct'].median(),
                    'total_return': results_df['total_return'].sum(),
                    'avg_win_rate': results_df['win_rate'].mean(),
                    'avg_sharpe_ratio': results_df['sharpe_ratio'].mean(),
                    'positive_performers_count': positive_performers,
                    'positive_performers_pct': positive_performers/len(results_df)*100,
                    'best_performer': results_df.loc[results_df['total_return_pct'].idxmax(), 'symbol'],
                    'best_performance': results_df['total_return_pct'].max(),
                    'worst_performer': results_df.loc[results_df['total_return_pct'].idxmin(), 'symbol'],
                    'worst_performance': results_df['total_return_pct'].min(),
                    'total_trades': total_trades,
                    'avg_trades_per_stock': avg_total_trades
                }
            }
        
        return {'individual_results': [], 'aggregate_metrics': {}}

def run_btst_backtest():
    """
    Run the BuyTodaySellTomorrow backtest on sample stocks
    """
    backtester = BuyTodaySellTomorrowBacktester(
        initial_capital=100000,
        max_positions=5,
        position_size_pct=0.1
    )
    
    # Sample of NSE stocks to test
    nse_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
        'LT.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS'
    ]
    
    # Run backtest for the past year
    results = backtester.backtest_universe(
        symbols=nse_stocks,
        start_date='2025-01-01',
        end_date='2026-01-01'
    )
    
    return results

if __name__ == "__main__":
    run_btst_backtest()