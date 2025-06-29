#!/bin/bash

# ML CSV Analyzer - Simple Web Deployment
echo "🚀 ML CSV Analyzer - Web Deployment"
echo "==================================="
echo ""

# Check if virtual environment exists
if [ ! -d "ml_analyzer_env" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python -m venv ml_analyzer_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ml_analyzer_env/bin/activate

# Check dependencies
if ! ml_analyzer_env/bin/python -c "import streamlit" &> /dev/null; then
    echo "❌ Dependencies not installed. Installing from requirements.txt..."
    ml_analyzer_env/bin/pip install --upgrade pip
    ml_analyzer_env/bin/pip install -r requirements.txt
fi

# Check ngrok
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found. Please install ngrok from https://ngrok.com/"
    exit 1
fi

# Check if users.csv exists, if not create default users
if [ ! -f "users.csv" ]; then
    echo "🔑 No users found. Creating default users..."
    python -c "from auth_utils import create_default_users; create_default_users()"
    echo "✅ Default users created:"
    echo "   • admin / admin123"
    echo "   • demo / demo123"
    echo "   • analyst / analyst123"
    echo ""
fi

echo "✅ All dependencies ready!"
echo ""

# Start Streamlit in background
echo "🚀 Starting Streamlit application..."
ml_analyzer_env/bin/streamlit run app.py --server.port=8501 --server.address=localhost --server.headless=true &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to initialize..."
sleep 8

# Check if Streamlit is running
if ! curl -s http://localhost:8501 > /dev/null 2>&1; then
    echo "❌ Streamlit failed to start. Check for errors above."
    kill $STREAMLIT_PID 2>/dev/null
    exit 1
fi

echo "✅ Streamlit is running on localhost:8501"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $STREAMLIT_PID 2>/dev/null
    pkill -f "ngrok http" 2>/dev/null
    echo "✅ Cleanup complete."
}

trap cleanup EXIT INT TERM

# Start ngrok tunnel
echo "🌐 Creating public web tunnel..."
echo ""
echo "📱 Your app will be accessible from anywhere on the internet!"
echo "🔗 ngrok will assign a random URL like: https://1234-5678-9abc.ngrok.io"
echo ""
echo "🔐 SECURITY INFO:"
echo "   • App has built-in login page"
echo "   • Demo credentials: demo/demo123"
echo "   • Users must login before accessing features"
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "   • URL changes each time you restart ngrok (free plan)"
echo "   • Free plan has 2-hour session limits"
echo "   • Your laptop must stay on and connected"
echo "   • Press Ctrl+C to stop everything"
echo ""
echo "💡 Manage users with: python manage_users.py"
echo ""

# Wait a moment then start ngrok
sleep 2
echo "🔗 Starting tunnel..."

# Start standard ngrok (free plan - no custom subdomain)
ngrok http 8501 