#!/bin/bash

# Fish Audio Transcription App Runner
echo "🎵 Starting Fish Audio Transcription App..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project directory."
    exit 1
fi

# Force use of Python 3.9 user environment (avoid system Python 3.10)
export PATH="/Users/ellelammba/Library/Python/3.9/bin:$PATH"
export PYTHONPATH="/Users/ellelammba/Library/Python/3.9/lib/python/site-packages:$PYTHONPATH"

echo "🔧 Setting up environment..."
echo "Using Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Ensure we have the right dependencies
echo "📦 Installing/checking dependencies..."
python3 -m pip install --user --quiet streamlit==1.39.0 requests ormsgpack 2>/dev/null || {
    echo "⚠️ Installing dependencies..."
    python3 -m pip install --user streamlit==1.39.0 requests ormsgpack
}

# Verify imports work
if python3 -c "import streamlit, requests, ormsgpack; print('✅ All imports successful')" 2>/dev/null; then
    echo "✅ Dependencies verified"
else
    echo "❌ Dependency verification failed"
    echo "Trying alternative installation..."
    python3 -m pip install --user --force-reinstall streamlit==1.39.0
fi

# Run the app
echo ""
echo "🚀 Starting Streamlit app..."
echo "📱 App will open at: http://localhost:8501"
echo "💡 Features: Large file support (76MB+), auto-compression, smart error handling"
echo ""

# Use the working environment directly
python3 -m streamlit run app.py --server.address localhost --server.port 8501 