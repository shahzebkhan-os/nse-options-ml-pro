# src/backtest/backtest_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BacktestEngine:
    def __init__(self, initial_capital=100000, transaction_cost=0.001):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital (₹100,000 default)
            transaction_cost: Cost per transaction (0.1% default)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.trades = []
        self.portfolio_values = []
        self.signals = []
        
    def backtest_strategy(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest the BuyTodaySellTomorrow strategy for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary containing backtest results
        """
        # Fetch historical data
        stock_data = yf.download(symbol, start=start_date, end=end_date, interval='1d')
        
        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Generate signals using technical indicators
        stock_data = self._generate_signals(stock_data)
        
        # Execute trades based on signals
        results = self._execute_trades(stock_data)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_metrics(results)
        
        return {
            'symbol': symbol,
            'performance': performance_metrics,
            'trades': self.trades,
            'portfolio_values': self.portfolio_values,
            'signals': self.signals
        }
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals using technical indicators
        """
        df = df.copy()
        
        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
        df['BB_Upper'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Generate buy/sell signals based on multiple criteria
        # Buy signal: RSI < 40 (oversold) AND price below SMA_20 AND volume above average
        df['Buy_Signal'] = (
            (df['RSI'] < 40) & 
            (df['Close'] < df['SMA_20']) & 
            (df['Volume'] > df['Volume_MA']) &
            (df['MACD'] > df['MACD_Signal'])  # Additional confirmation
        )
        
        # Sell signal: RSI > 70 (overbought) OR price above upper Bollinger Band
        df['Sell_Signal'] = (
            (df['RSI'] > 70) | 
            (df['Close'] > df['BB_Upper'])
        )
        
        # Shift signals by 1 day to simulate next-day execution
        df['Buy_Signal_Next'] = df['Buy_Signal'].shift(1)
        df['Sell_Signal_Next'] = df['Sell_Signal'].shift(1)
        
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
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    def _execute_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute trades based on generated signals
        """
        cash = self.initial_capital
        shares_held = 0
        current_position = None
        position_entry_price = 0
        position_entry_date = None
        
        portfolio_value = self.initial_capital
        
        for idx, row in df.iterrows():
            # Record portfolio value
            if shares_held > 0:
                portfolio_value = cash + (shares_held * row['Close'])
            else:
                portfolio_value = cash
                
            self.portfolio_values.append({
                'date': idx.date(),
                'portfolio_value': portfolio_value,
                'cash': cash,
                'shares_held': shares_held,
                'current_price': row['Close']
            })
            
            # Check for sell signal
            if current_position == 'long' and row['Sell_Signal_Next']:
                # Sell position
                proceeds = shares_held * row['Close']
                transaction_fee = proceeds * self.transaction_cost
                cash += proceeds - transaction_fee
                
                # Record trade
                trade_profit = (row['Close'] - position_entry_price) * shares_held
                self.trades.append({
                    'symbol': df.name if hasattr(df, 'name') else 'Unknown',
                    'entry_date': position_entry_date,
                    'exit_date': idx.date(),
                    'entry_price': position_entry_price,
                    'exit_price': row['Close'],
                    'shares': shares_held,
                    'profit_loss': trade_profit,
                    'return_pct': (row['Close'] - position_entry_price) / position_entry_price * 100
                })
                
                # Reset position
                shares_held = 0
                current_position = None
                position_entry_price = 0
                position_entry_date = None
            
            # Check for buy signal and no current position
            elif current_position is None and row['Buy_Signal_Next']:
                # Calculate position size (risk management)
                position_size = min(cash * 0.1, cash)  # Max 10% of portfolio per trade
                shares_to_buy = int(position_size // row['Close'])
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['Close']
                    transaction_fee = cost * self.transaction_cost
                    total_cost = cost + transaction_fee
                    
                    if total_cost <= cash:
                        cash -= total_cost
                        shares_held = shares_to_buy
                        current_position = 'long'
                        position_entry_price = row['Close']
                        position_entry_date = idx.date()
        
        return df
    
    def _calculate_metrics(self, results: Dict) -> Dict:
        """
        Calculate performance metrics
        """
        if not self.trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_return_per_trade': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'profit_factor': 0
            }
        
        # Convert trades to DataFrame for easier calculations
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit_loss'] > 0])
        losing_trades = len(trades_df[trades_df['profit_loss'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate returns
        total_return = trades_df['profit_loss'].sum()
        avg_return_per_trade = trades_df['profit_loss'].mean()
        
        # Calculate portfolio value changes over time for advanced metrics
        portfolio_df = pd.DataFrame(self.portfolio_values)
        
        # Max drawdown calculation
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max']
        max_drawdown = abs(portfolio_df['drawdown'].min()) if not portfolio_df.empty else 0
        
        # Return percentage calculation
        start_portfolio = self.initial_capital
        end_portfolio = portfolio_df['portfolio_value'].iloc[-1] if not portfolio_df.empty else self.initial_capital
        total_return_pct = (end_portfolio - start_portfolio) / start_portfolio * 100
        
        # Sharpe ratio calculation (assuming risk-free rate of 6% annually)
        if len(portfolio_df) > 1:
            daily_returns = portfolio_df['portfolio_value'].pct_change().dropna()
            excess_returns = daily_returns - (0.06 / 252)  # Daily risk-free rate
            if excess_returns.std() != 0:
                sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Sortino ratio (using downside deviation)
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = negative_returns.std()
                if downside_deviation != 0:
                    sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252)
                else:
                    sortino_ratio = 0
            else:
                sortino_ratio = sharpe_ratio  # If no negative returns, use sharpe ratio
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Profit factor calculation
        gross_profits = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
        gross_losses = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        return {
            'total_return': round(total_return, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate * 100, 2),
            'avg_return_per_trade': round(avg_return_per_trade, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'profit_factor': round(profit_factor, 2) if isinstance(profit_factor, (int, float)) and not np.isinf(profit_factor) else 'Inf'
        }

def run_comprehensive_backtest():
    """
    Run comprehensive backtest for multiple stocks
    """
    backtester = BacktestEngine(initial_capital=100000)
    
    # Define NSE stocks to backtest (top Nifty 50 stocks)
    nse_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS',
        'LT.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'TITAN.NS',
        'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'GRASIM.NS', 'BAJFINANCE.NS'
    ]
    
    # Define backtest period (last 2 years)
    start_date = '2024-01-01'
    end_date = '2026-01-01'
    
    results_summary = []
    
    print("Starting comprehensive backtest...")
    print(f"Testing {len(nse_stocks)} stocks from {start_date} to {end_date}")
    print("="*60)
    
    for i, stock in enumerate(nse_stocks):
        print(f"Backtesting {stock} ({i+1}/{len(nse_stocks)})...")
        
        try:
            result = backtester.backtest_strategy(stock, start_date, end_date)
            
            if 'error' not in result:
                perf = result['performance']
                results_summary.append({
                    'symbol': stock,
                    'total_return': perf['total_return'],
                    'total_return_pct': perf['total_return_pct'],
                    'total_trades': perf['total_trades'],
                    'win_rate': perf['win_rate'],
                    'sharpe_ratio': perf['sharpe_ratio'],
                    'max_drawdown': perf['max_drawdown']
                })
                
                print(f"  ✓ {stock}: {perf['total_return_pct']:.2f}% return, "
                      f"{perf['win_rate']:.1f}% win rate, {perf['sharpe_ratio']:.2f} SR")
            else:
                print(f"  ✗ {stock}: {result['error']}")
                
        except Exception as e:
            print(f"  ✗ {stock}: Error - {str(e)}")
    
    # Calculate aggregate metrics
    if results_summary:
        agg_df = pd.DataFrame(results_summary)
        
        print("\n" + "="*60)
        print("AGGREGATE RESULTS")
        print("="*60)
        print(f"Total stocks tested: {len(agg_df)}")
        print(f"Average return: {agg_df['total_return_pct'].mean():.2f}%")
        print(f"Median return: {agg_df['total_return_pct'].median():.2f}%")
        print(f"Avg win rate: {agg_df['win_rate'].mean():.2f}%")
        print(f"Avg Sharpe ratio: {agg_df['sharpe_ratio'].mean():.2f}")
        print(f"Avg max drawdown: {agg_df['max_drawdown'].mean():.2f}%")
        print(f"Best performer: {agg_df.loc[agg_df['total_return_pct'].idxmax(), 'symbol']} "
              f"({agg_df['total_return_pct'].max():.2f}%)")
        print(f"Worst performer: {agg_df.loc[agg_df['total_return_pct'].idxmin(), 'symbol']} "
              f"({agg_df['total_return_pct'].min():.2f}%)")
        
        # Count positive performers
        positive_count = len(agg_df[agg_df['total_return_pct'] > 0])
        print(f"Positive performers: {positive_count}/{len(agg_df)} "
              f"({positive_count/len(agg_df)*100:.1f}%)")
    
    return results_summary

if __name__ == "__main__":
    run_comprehensive_backtest()