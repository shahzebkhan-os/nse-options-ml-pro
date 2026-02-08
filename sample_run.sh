#!/bin/bash
echo "1. Installing Dependencies..."
python3 -m pip install -r requirements.txt

echo "2. Launching NIFTY AI Dashboard..."
# The Dashboard now handles training internally
python3 -m streamlit run src/ui/app.py
