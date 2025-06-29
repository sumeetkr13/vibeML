#!/bin/bash

# Simple test to verify the app works locally
echo "🧪 Testing ML CSV Analyzer Locally"
echo "================================"
echo ""

# Activate virtual environment
source ml_analyzer_env/bin/activate

# Start Streamlit
echo "🚀 Starting Streamlit on localhost:8501..."
echo "🌐 Open your browser to: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop"
echo ""

ml_analyzer_env/bin/streamlit run app.py --server.port=8501 --server.address=localhost 