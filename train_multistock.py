import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from src.ingest.connector import DataConnector
from src.features.indicators import compute_indicators
from src.models.temporal import LSTMModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

STOCKS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
          "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]

def prepare_data_for_stock(symbol, lookback=45):
    connector = DataConnector()
    df = connector.fetch_ohlcv(symbol, period="2y")
    if df is None or df.empty: return None, None, None
    
    # Features
    df = compute_indicators(df)
    df['Log_Volume'] = np.log(df['Volume'] + 1)
    
    features = ['Close', 'RSI', 'MACD', 'ATR', 'Log_Return', 'Log_Volume']
    df = df.dropna()
    
    data = df[features].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - lookback - 1):
        X.append(data_scaled[i : i+lookback])
        current_close = df['Close'].iloc[i + lookback - 1]
        next_close = df['Close'].iloc[i + lookback]
        target = 1.0 if float(next_close) > float(current_close) else 0.0
        y.append(target)
        
    return np.array(X), np.array(y).reshape(-1, 1), scaler

def train_general_model():
    logger.info("Starting Multi-Stock Training...")
    
    X_all, y_all = [], []
    
    # 1. Collect Data from 9 stocks (Hold out RELIANCE for testing)
    train_stocks = [s for s in STOCKS if s != "RELIANCE"]
    
    for symbol in train_stocks:
        logger.info(f"Processing {symbol}...")
        X, y, _ = prepare_data_for_stock(symbol)
        if X is not None:
            X_all.append(X)
            y_all.append(y)
            
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    
    # 2. Train Model
    input_dim = X_train.shape[2]
    model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=2, dropout=0.3)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    logger.info(f"Training on {len(X_train)} combined samples...")
    epochs = 50
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
            
    return model

def test_on_target(model):
    logger.info("Testing on Held-out Stock: RELIANCE")
    X, y, _ = prepare_data_for_stock("RELIANCE")
    
    X_test_t = torch.FloatTensor(X)
    y_test_t = torch.FloatTensor(y)
    
    model.eval()
    with torch.no_grad():
        out = model(X_test_t)
        predicted = (out > 0.5).float()
        accuracy = (predicted == y_test_t).float().mean()
        logger.info(f"General Model Accuracy on RELIANCE: {accuracy.item() * 100:.2f}%")

if __name__ == "__main__":
    model = train_general_model()
    test_on_target(model)
