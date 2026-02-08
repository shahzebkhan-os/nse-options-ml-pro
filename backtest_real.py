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

def prepare_sequences(df, lookback=45):
    # Features: Reduced and focused
    # Add Log Volume
    df['Log_Volume'] = np.log(df['Volume'] + 1)
    
    features = [
        'Close', 'RSI', 'MACD', 
        'ATR', 'Log_Return', 'Log_Volume'
    ]
    df = df.dropna()
    
    data = df[features].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    
    for i in range(len(data_scaled) - lookback - 1):
        X.append(data_scaled[i : i+lookback])
        
        current_close = df['Close'].iloc[i + lookback - 1]
        next_close = df['Close'].iloc[i + lookback]
        
        # Target: 1 if Next Close > Current Close
        target = 1.0 if float(next_close) > float(current_close) else 0.0
        y.append(target)
        
    return np.array(X), np.array(y).reshape(-1, 1), scaler

def backtest(symbol="RELIANCE"):
    logger.info(f"Starting Backtest/Backtrack for {symbol}...")
    
    # 1. Fetch Data
    connector = DataConnector()
    df = connector.fetch_ohlcv(symbol, period="2y") # 2 years of data
    if df is None or df.empty:
        logger.error("No data fetched.")
        return
        
    # 2. Features
    df = compute_indicators(df)
    
    # 3. Prepare Data
    lookback = 45
    X, y, scaler = prepare_sequences(df, lookback)
    
    # Split Train (80%) / Test (20%)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    # 4. Model Setup
    input_dim = X_train.shape[2]
    # Added dropout=0.3
    model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.3)
    criterion = nn.BCELoss()
    # Lower LR slightly to learning stable
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # 5. Train
    logger.info(f"Training on {len(X_train)} samples...")
    epochs = 100 # Increased epochs since we lowered LR
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
            
    # 6. Evaluate (Backtrack)
    logger.info("Evaluating on Test Data (Backtracking)...")
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_t)
        predicted = (test_out > 0.5).float()
        
        accuracy = (predicted == y_test_t).float().mean()
        logger.info(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
        
        # Calculate Directional Accuracy
        correct_directions = (predicted == y_test_t).sum().item()
        total_predictions = len(y_test_t)
        logger.info(f"Directional Correctness: {correct_directions}/{total_predictions}")

if __name__ == "__main__":
    backtest("RELIANCE")
