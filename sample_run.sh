#!/bin/bash
echo "1. Installing Dependencies..."
# Ensure pip is installed and install requirements
python3 -m pip install -r requirements.txt

echo "2. Running Training Demo..."
# Use python3 explicitely
python3 train.py --symbol RELIANCE --epochs 2

echo "3. Starting Pipeline Simulation (Background)..."
python3 train.py --demo &
PID=$!

echo "4. Launching UI..."
# Run streamlit via python module to avoid path issues
python3 -m streamlit run src/ui/app.py

kill $PID
