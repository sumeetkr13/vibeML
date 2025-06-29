#!/bin/bash

# ML CSV Analyzer - Local Runner
echo "🚀 Starting ML CSV Analyzer..."
echo "📊 This will open the app in your default browser"
echo ""

# Check if virtual environment exists
if [ ! -d "ml_analyzer_env" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python -m venv ml_analyzer_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ml_analyzer_env/bin/activate

# Check if streamlit is installed in the virtual environment
if ! ml_analyzer_env/bin/python -c "import streamlit" &> /dev/null; then
    echo "❌ Dependencies not installed. Installing from requirements.txt..."
    ml_analyzer_env/bin/pip install --upgrade pip
    ml_analyzer_env/bin/pip install -r requirements.txt
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ app.py not found. Make sure you're in the correct directory."
    exit 1
fi

echo "✅ Starting Streamlit application..."
echo "🌐 The app will open at: http://localhost:8501"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

# Run the streamlit app using the virtual environment's python
# Use explicit local settings for development
ml_analyzer_env/bin/streamlit run app.py --server.port=8501 --server.address=localhost 