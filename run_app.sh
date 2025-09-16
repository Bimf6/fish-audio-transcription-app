#!/bin/bash

# Fish Audio Transcription App Runner
echo "🎵 Starting Fish Audio Transcription App..."

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project directory."
    exit 1
fi

# Install/update dependencies with fallback
echo "📦 Installing dependencies..."

# Try to install core dependencies with fallback versions
echo "Installing core packages..."
python3 -m pip install --user streamlit requests ormsgpack --quiet 2>/dev/null || {
    echo "⚠️  Installing with older versions for compatibility..."
    python3 -m pip install --user "streamlit==1.39.0" "requests>=2.28.0" "ormsgpack>=1.4.0" --quiet
}

# Run the app
echo "🚀 Starting Streamlit app..."
echo "📱 App will open at: http://localhost:8501"
echo "💡 Features: Large file support (76MB+), auto-compression, smart error handling"
echo ""

# Use the Python version that has working Streamlit (3.9 user install)
echo "Using Python 3 with user packages..."
export PATH="/Users/ellelammba/Library/Python/3.9/bin:$PATH"
python3 -m streamlit run app.py 