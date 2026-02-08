import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.models.temporal import LSTMModel
from src.pipeline.worker import Worker
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="RELIANCE")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    connector = DataConnector()
    
    if args.demo:
        # Run Pipeline Simulation
        logger.info("Starting Pipeline Demo...")
        symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"] * 5
        worker = Worker(symbols)
        worker.process()
        return

    # Data
    logger.info(f"Training on {args.symbol}...")
    df = connector.fetch_ohlcv(args.symbol)
    if df is None: return

    # Features
    df = compute_indicators(df)
    
    # Mock Tensor for demo training
    X = torch.randn(100, 10, 14) # batch, seq, feature
    y = torch.randint(0, 2, (100, 1)).float()
    
    # Model
    model = LSTMModel(input_dim=14, hidden_dim=32)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loop
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch {epoch+1}/{args.epochs} Loss: {loss.item():.4f}")
        
    torch.save(model.state_dict(), "model.pth")
    logger.info("Model saved.")

if __name__ == "__main__":
    main()
