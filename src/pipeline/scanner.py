import concurrent.futures
from src.pipeline.predictor import VolatilityPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ParallelScanner:
    def __init__(self, predictor: VolatilityPredictor):
        self.predictor = predictor

    def scan_list(self, stock_list, max_workers=5):
        """Scans a list of stocks in parallel."""
        results = []
        
        # Ensure model is trained BEFORE threading to avoid race conditions
        if not self.predictor.is_trained:
            logger.info("Pre-training models before parallel scan...")
            self.predictor.train_sector_models()
            self.predictor.is_trained = True

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map symbol to future
            future_to_stock = {executor.submit(self.predictor.predict, stock): stock for stock in stock_list}
            
            for future in concurrent.futures.as_completed(future_to_stock):
                stock = future_to_stock[future]
                try:
                    data = future.result()
                    if data:
                        results.append(data)
                except Exception as e:
                    logger.error(f"Parallel scan failed for {stock}: {e}")
                    
        return results
