import time
import random
from src.pipeline.eta import ETACalculator
from src.utils.logger import get_logger

logger = get_logger(__name__)

class Worker:
    def __init__(self, items):
        self.items = items
        self.eta = ETACalculator()
        self.progress = {}

    def process(self):
        total = len(self.items)
        for i, item in enumerate(self.items):
            # Simulate work
            time.sleep(random.uniform(0.1, 0.5))
            
            self.eta.update(1)
            remaining = total - (i + 1)
            eta_str = self.eta.get_eta(remaining)
            
            logger.info(f"Processed {item} ({i+1}/{total}). ETA: {eta_str}")
            self.progress[item] = "Done"
            
            # Hook for UI update (simplified via shared dict/file in real app)
            with open("progress_status.txt", "w") as f:
                f.write(f"{i+1}/{total}|{eta_str}|{item}")
