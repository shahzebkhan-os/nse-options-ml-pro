#!/bin/bash
echo "1. Installing Dependencies..."
# pip install -r requirements.txt

echo "2. Running Training Demo..."
python train.py --symbol RELIANCE --epochs 2

echo "3. Starting Pipeline Simulation (Background)..."
python train.py --demo &
PID=$!

echo "4. Launching UI..."
streamlit run src/ui/app.py

kill $PID
