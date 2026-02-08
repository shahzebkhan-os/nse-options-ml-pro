from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators

c = DataConnector()
df = c.fetch_ohlcv("RELIANCE", period="1y")
print("Original DF shape:", df.shape)
print(df.head())
print("Columns:", df.columns)

df = compute_indicators(df)
if df is not None:
    print("Processed DF shape:", df.shape)
    print(df.tail())
else:
    print("Processed DF is None")
