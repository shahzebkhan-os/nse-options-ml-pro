import pytest
import torch
from src.models.temporal import LSTMModel
from src.options.bs_pricing import BSModel

def test_lstm_forward():
    model = LSTMModel(input_dim=10, hidden_dim=16)
    x = torch.randn(5, 20, 10) # Batch, Seq, Feat
    out = model(x)
    assert out.shape == (5, 1)

def test_bs_call_price():
    # S=100, K=100, T=1, r=0.05, sigma=0.2
    # Known ~10.45
    price = BSModel.call_price(100, 100, 1, 0.05, 0.2)
    assert 10.0 < price < 11.0
