# How ETA is Computed

We use an **Exponential Moving Average (EMA)** to estimate the time remaining for the pipeline.

## Formula

The average time per item $t_{avg}$ is updated as:

$$ t_{avg} = \alpha \cdot t_{current} + (1 - \alpha) \cdot t_{avg} $$

Where:
- $\alpha$: Smoothing factor (default 0.3)
- $t_{current}$: Time taken for the latest batch / batch size.

## Code

```python
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
```
