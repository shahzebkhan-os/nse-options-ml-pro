import concurrent.futures
from src.pipeline.predictor import VolatilityPredictor
from src.utils.stock_lists import ALL_STOCKS
import threading
from typing import List, Dict, Any
import time

class ParallelStockScanner:
    def __init__(self, max_workers=8):
        """
        Initializes the parallel stock scanner.
        
        Args:
            max_workers (int): Number of parallel workers to use for scanning
        """
        self.max_workers = max_workers
        self.results = {}
        self.lock = threading.Lock()
        
    def _analyze_single_stock(self, stock_symbol: str) -> Dict[str, Any]:
        """
        Analyzes a single stock and returns the prediction results.
        
        Args:
            stock_symbol (str): The stock symbol to analyze
            
        Returns:
            Dict[str, Any]: Prediction results for the stock
        """
        try:
            # Create a predictor instance for this thread
            predictor = VolatilityPredictor()
            result = predictor.predict(stock_symbol)
            
            if result:
                # Format the result for the scanner
                formatted_result = {
                    "Symbol": result["Symbol"],
                    "Price": f"₹{result['Live_Price']:.2f}",
                    "Change": f"{result['Pct_Change']:.2f}%",
                    "Regime": result["Regime"],
                    "Conf": result["Confidence"],
                    "PCR": result["Options"]["PCR_OI"] if result["Options"] else 0,
                    "Dir 1D": result["Directional_Predictions"].get("1d", {}).get("direction", "N/A") if "Directional_Predictions" in result else "N/A",
                    "Dir Conf 1D": result["Directional_Predictions"].get("1d", {}).get("confidence", "N/A") if "Directional_Predictions" in result else "N/A",
                    "raw_result": result  # Keep raw result for detailed analysis
                }
                return formatted_result
            else:
                # Return empty result if prediction failed
                return {
                    "Symbol": stock_symbol,
                    "Price": "N/A",
                    "Change": "N/A",
                    "Regime": "N/A",
                    "Conf": "N/A",
                    "PCR": "N/A",
                    "Dir 1D": "N/A",
                    "Dir Conf 1D": "N/A",
                    "raw_result": None
                }
        except Exception as e:
            print(f"Error analyzing {stock_symbol}: {e}")
            return {
                "Symbol": stock_symbol,
                "Price": "N/A",
                "Change": "N/A",
                "Regime": "N/A",
                "Conf": "N/A",
                "PCR": "N/A",
                "Dir 1D": "N/A",
                "Dir Conf 1D": "N/A",
                "raw_result": None
            }
    
    def scan_stocks(self, stock_list: List[str], callback=None) -> List[Dict[str, Any]]:
        """
        Scans a list of stocks in parallel and returns the results.
        
        Args:
            stock_list (List[str]): List of stock symbols to scan
            callback (callable, optional): Function to call with intermediate results
            
        Returns:
            List[Dict[str, Any]]: List of formatted results for each stock
        """
        results = []
        
        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_stock = {executor.submit(self._analyze_single_stock, stock): stock 
                              for stock in stock_list}
            
            # Process completed tasks as they finish (real-time results)
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Call the callback with intermediate results if provided
                    if callback:
                        callback(result, len(results), len(stock_list))
                        
                except Exception as e:
                    print(f"Error processing {stock}: {e}")
                    error_result = {
                        "Symbol": stock,
                        "Price": "N/A",
                        "Change": "N/A",
                        "Regime": "N/A",
                        "Conf": "N/A",
                        "PCR": "N/A",
                        "Dir 1D": "N/A",
                        "Dir Conf 1D": "N/A",
                        "raw_result": None
                    }
                    results.append(error_result)
        
        # Sort results by symbol for consistency
        results.sort(key=lambda x: x["Symbol"])
        return results

def scan_watchlist(watchlist: List[str], scan_limit: int = 10):
    """
    Convenience function to scan a watchlist of stocks.
    
    Args:
        watchlist (List[str]): List of stock symbols to scan
        scan_limit (int): Maximum number of stocks to scan
        
    Returns:
        List[Dict[str, Any]]: Scan results
    """
    scanner = ParallelStockScanner(max_workers=8)  # Use 8 parallel workers
    
    # Limit the watchlist to the specified number
    limited_watchlist = watchlist[:scan_limit]
    
    def progress_callback(result, completed, total):
        """Callback to show progress during scanning."""
        print(f"Scanned {completed}/{total}: {result['Symbol']} - {result['Regime']}")
    
    results = scanner.scan_stocks(limited_watchlist, callback=progress_callback)
    return results

if __name__ == "__main__":
    # Example usage
    combined_list = ALL_STOCKS["Indices (NIFTY/BANKNIFTY/SENSEX)"] + ALL_STOCKS["Large Cap (Nifty 50 Stocks)"]
    results = scan_watchlist(combined_list, scan_limit=10)
    
    for result in results:
        print(f"{result['Symbol']}: {result['Regime']} - {result['Dir 1D']}")