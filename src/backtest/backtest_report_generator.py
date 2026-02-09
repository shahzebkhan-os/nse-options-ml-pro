# src/backtest/backtest_report_generator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

class BacktestReportGenerator:
    def __init__(self):
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_performance_report(self, backtest_results: dict, output_path: str = "backtest_report.html"):
        """
        Generate comprehensive HTML report of backtest results
        """
        # Create the report
        html_content = self.create_html_report(backtest_results)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Backtest report saved to {output_path}")
        return output_path
    
    def create_html_report(self, results: dict) -> str:
        """Create HTML report content"""
        agg_metrics = results['aggregate_metrics']
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>BuyTodaySellTomorrow Backtest Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }}
                .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                .metric-label {{ font-size: 14px; color: #6c757d; }}
                .chart-container {{ margin: 30px 0; }}
                .table-container {{ overflow-x: auto; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 BuyTodaySellTomorrow Backtest Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>📊 Aggregate Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{agg_metrics['avg_return']:.2f}%</div>
                        <div class="metric-label">Avg Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{agg_metrics['avg_win_rate']:.2f}%</div>
                        <div class="metric-label">Avg Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{agg_metrics['avg_sharpe_ratio']:.2f}</div>
                        <div class="metric-label">Avg Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{agg_metrics['positive_performers_pct']:.1f}%</div>
                        <div class="metric-label">Positive Performers</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{agg_metrics['total_trades']}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{agg_metrics['total_stocks']}</div>
                        <div class="metric-label">Total Stocks</div>
                    </div>
                </div>
                
                <h2>📈 Individual Stock Performance</h2>
                <div class="table-container">
                    {self.create_performance_table(results['individual_results'])}
                </div>
                
                <h2>🏆 Performance Distribution</h2>
                <div id="performance_dist" class="chart-container" style="height: 500px;"></div>
                
                <h2>📊 Win Rate vs Return</h2>
                <div id="winrate_return" class="chart-container" style="height: 500px;"></div>
                
                <h2>🏆 Top Performers</h2>
                <div class="table-container">
                    <table>
                        <tr><th>Rank</th><th>Symbol</th><th>Total Return (%)</th><th>Win Rate (%)</th><th>Sharpe Ratio</th></tr>
                        {self.create_top_performers_table(results['individual_results'])}
                    </table>
                </div>
                
                <h2>⚠️ Risk Analysis</h2>
                <div class="table-container">
                    <table>
                        <tr><th>Symbol</th><th>Max Drawdown (%)</th><th>Profit Factor</th><th>Best Trade (%)</th><th>Worst Trade (%)</th></tr>
                        {self.create_risk_table(results['individual_results'])}
                    </table>
                </div>
                
                <script>
                    // Performance Distribution Chart
                    var perf_data = {self.create_performance_distribution_data(results['individual_results'])};
                    var perf_layout = {{
                        title: 'Return Distribution Across Stocks',
                        xaxis: {{title: 'Total Return (%)'}},
                        yaxis: {{title: 'Frequency'}}
                    }};
                    Plotly.newPlot('performance_dist', perf_data, perf_layout);
                    
                    // Win Rate vs Return Scatter
                    var scatter_data = {self.create_scatter_data(results['individual_results'])};
                    var scatter_layout = {{
                        title: 'Win Rate vs Total Return',
                        xaxis: {{title: 'Win Rate (%)'}},
                        yaxis: {{title: 'Total Return (%)'}}
                    }};
                    Plotly.newPlot('winrate_return', scatter_data, scatter_layout);
                </script>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def create_performance_table(self, results: list) -> str:
        """Create HTML table for performance metrics"""
        html = '<table><tr><th>Symbol</th><th>Total Return (%)</th><th>Win Rate (%)</th><th>Total Trades</th><th>Sharpe Ratio</th><th>Max Drawdown (%)</th></tr>'
        
        for result in results:
            return_class = "positive" if result['total_return_pct'] > 0 else "negative"
            html += f"""
            <tr>
                <td>{result['symbol']}</td>
                <td class="{return_class}">{result['total_return_pct']:.2f}%</td>
                <td>{result['win_rate']:.2f}%</td>
                <td>{result['total_trades']}</td>
                <td>{result['sharpe_ratio']:.2f}</td>
                <td>{result['max_drawdown']:.2f}%</td>
            </tr>
            """
        
        html += '</table>'
        return html
    
    def create_top_performers_table(self, results: list) -> str:
        """Create HTML table for top performers"""
        sorted_results = sorted(results, key=lambda x: x['total_return_pct'], reverse=True)[:10]
        html = ''
        
        for i, result in enumerate(sorted_results, 1):
            return_class = "positive" if result['total_return_pct'] > 0 else "negative"
            html += f"""
            <tr>
                <td>{i}</td>
                <td>{result['symbol']}</td>
                <td class="{return_class}">{result['total_return_pct']:.2f}%</td>
                <td>{result['win_rate']:.2f}%</td>
                <td>{result['sharpe_ratio']:.2f}</td>
            </tr>
            """
        
        return html
    
    def create_risk_table(self, results: list) -> str:
        """Create HTML table for risk metrics"""
        html = ''
        
        for result in results:
            html += f"""
            <tr>
                <td>{result['symbol']}</td>
                <td>{result['max_drawdown']:.2f}%</td>
                <td>{result['profit_factor']}</td>
                <td class="positive">{result['best_trade']:.2f}%</td>
                <td class="negative">{result['worst_trade']:.2f}%</td>
            </tr>
            """
        
        return html
    
    def create_performance_distribution_data(self, results: list) -> str:
        """Create data for performance distribution chart"""
        returns = [r['total_return_pct'] for r in results]
        hist_data = np.histogram(returns, bins=20)
        
        data_json = f"""[{{
            x: {returns},
            type: 'histogram',
            nbinsx: 20,
            marker: {{
                color: 'rgba(0, 123, 255, 0.7)',
                line: {{
                    color: 'rgba(0, 0, 0, 1)',
                    width: 1
                }}
            }}
        }}]"""
        
        return data_json
    
    def create_scatter_data(self, results: list) -> str:
        """Create data for scatter plot"""
        x_values = [r['win_rate'] for r in results]
        y_values = [r['total_return_pct'] for r in results]
        symbols = [r['symbol'] for r in results]
        
        data_json = f"""[{{
            x: {x_values},
            y: {y_values},
            mode: 'markers',
            text: {symbols},
            marker: {{
                size: 10,
                color: 'rgba(0, 123, 255, 0.7)',
                line: {{
                    color: 'rgba(0, 0, 0, 1)',
                    width: 1
                }}
            }},
            type: 'scatter'
        }}]"""
        
        return data_json
    
    def plot_equity_curve(self, trade_log: list, title: str = "Equity Curve"):
        """
        Plot equity curve from trade log
        """
        if not trade_log:
            print("No trade data to plot")
            return
        
        # Create a timeline of trades
        df = pd.DataFrame(trade_log)
        df['entry_date'] = pd.to_datetime(df['entry_date'])
        df['exit_date'] = pd.to_datetime(df['exit_date'])
        
        # Calculate cumulative returns
        df['cumulative_return'] = df['pnl'].cumsum()
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['exit_date'], df['cumulative_return'], linewidth=2, color='#007bff')
        ax.fill_between(df['exit_date'], df['cumulative_return'], alpha=0.3, color='#007bff')
        ax.set_title(f'{title} - Cumulative Returns Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (₹)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_trade_analysis(self, trade_log: list):
        """
        Create various plots for trade analysis
        """
        if not trade_log:
            print("No trade data to analyze")
            return
        
        df = pd.DataFrame(trade_log)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of returns
        axes[0, 0].hist(df['return_pct'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Trade Returns (%)')
        axes[0, 0].set_xlabel('Return (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(df['return_pct'].mean(), color='red', linestyle='--', label=f'Mean: {df["return_pct"].mean():.2f}%')
        axes[0, 0].legend()
        
        # 2. Win/Loss distribution
        wins = len(df[df['pnl'] > 0])
        losses = len(df[df['pnl'] < 0])
        axes[0, 1].pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%', colors=['#28a745', '#dc3545'])
        axes[0, 1].set_title('Win/Loss Distribution')
        
        # 3. P&L over time
        df_sorted = df.sort_values('exit_date')
        axes[1, 0].plot(range(len(df_sorted)), df_sorted['pnl'], marker='o', linestyle='', alpha=0.6)
        axes[1, 0].set_title('Individual Trade P&L')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('P&L (₹)')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Return vs Confidence Score
        if 'confidence_score' in df.columns:
            axes[1, 1].scatter(df['confidence_score'], df['return_pct'], alpha=0.6, color='purple')
            axes[1, 1].set_title('Return vs Confidence Score')
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Return (%)')
        
        plt.tight_layout()
        plt.show()
    
    def generate_statistical_summary(self, results: dict):
        """
        Generate statistical summary of backtest results
        """
        print("="*60)
        print("BACKTEST STATISTICAL SUMMARY")
        print("="*60)
        
        agg = results['aggregate_metrics']
        
        print(f"Total Stocks Tested: {agg['total_stocks']}")
        print(f"Average Return: {agg['avg_return']:.2f}%")
        print(f"Median Return: {agg['median_return']:.2f}%")
        print(f"Total Return: {agg['total_return']:,.2f}₹")
        print(f"Average Win Rate: {agg['avg_win_rate']:.2f}%")
        print(f"Average Sharpe Ratio: {agg['avg_sharpe_ratio']:.2f}")
        print(f"Positive Performers: {agg['positive_performers_count']}/{agg['total_stocks']} ({agg['positive_performers_pct']:.1f}%)")
        print(f"Total Trades: {agg['total_trades']}")
        print(f"Average Trades per Stock: {agg['avg_trades_per_stock']:.1f}")
        print(f"Best Performer: {agg['best_performer']} ({agg['best_performance']:.2f}%)")
        print(f"Worst Performer: {agg['worst_performer']} ({agg['worst_performance']:.2f}%)")
        
        # Calculate additional statistics
        individual_returns = [r['total_return_pct'] for r in results['individual_results']]
        volatility = np.std(individual_returns)
        print(f"Return Volatility: {volatility:.2f}%")
        
        # Information ratio (excess return per unit of active risk)
        if volatility != 0:
            info_ratio = agg['avg_return'] / volatility
            print(f"Information Ratio: {info_ratio:.2f}")
        
        print("="*60)

def run_complete_backtest_analysis():
    """
    Run complete backtest analysis with reports
    """
    from src.backtest.buy_today_sell_tomorrow_backtest import BuyTodaySellTomorrowBacktester
    
    # Initialize backtester
    backtester = BuyTodaySellTomorrowBacktester(
        initial_capital=100000,
        max_positions=5,
        position_size_pct=0.1
    )
    
    # Sample of NSE stocks for testing
    nse_stocks = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS'
    ]
    
    print("Running comprehensive backtest analysis...")
    
    # Run backtest
    results = backtester.backtest_universe(
        symbols=nse_stocks,
        start_date='2025-06-01',
        end_date='2026-01-01'
    )
    
    # Generate report
    report_gen = BacktestReportGenerator()
    
    # Print statistical summary
    report_gen.generate_statistical_summary(results)
    
    # Generate HTML report
    report_file = report_gen.generate_performance_report(results, "btst_backtest_report.html")
    
    print(f"\nDetailed analysis complete!")
    print(f"HTML Report: {report_file}")
    
    # If we have trade logs, generate plots for the first stock
    if results['individual_results']:
        first_symbol = results['individual_results'][0]['symbol']
        print(f"\nGenerating detailed plots for {first_symbol}...")
        
        # Get the trade log for the first stock
        for result in backtester.trade_log:
            if result['symbol'] == first_symbol:
                # Create plots
                report_gen.plot_equity_curve(backtester.trade_log, f"Equity Curve - {first_symbol}")
                report_gen.plot_trade_analysis(backtester.trade_log)
                break
    
    return results, report_file

if __name__ == "__main__":
    run_complete_backtest_analysis()