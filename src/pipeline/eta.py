import time
from collections import deque

class ETACalculator:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.avg_time_per_item = 0
        self.last_update = time.time()
        self.history = deque(maxlen=10)

    def update(self, num_processed):
        now = time.time()
        elapsed = now - self.last_update
        if num_processed > 0:
            current_rate = elapsed / num_processed
            if self.avg_time_per_item == 0:
                self.avg_time_per_item = current_rate
            else:
                # EMA update
                self.avg_time_per_item = (self.alpha * current_rate) + ((1 - self.alpha) * self.avg_time_per_item)
        
        self.last_update = now

    def get_eta(self, remaining_items):
        if self.avg_time_per_item == 0:
            return "Calculating..."
        seconds = remaining_items * self.avg_time_per_item
        return time.strftime("%H:%M:%S", time.gmtime(seconds))
