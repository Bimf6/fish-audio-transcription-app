#!/bin/bash

# Simple alternative runner for Fish Audio Transcription App
echo "🎵 Fish Audio Transcription - Simple Runner"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the project directory."
    exit 1
fi

echo "🔍 Testing Python environments..."

# Method 1: Try with user Python 3.9 packages
export PATH="/Users/ellelammba/Library/Python/3.9/bin:$PATH"
if python3 -c "import streamlit, requests, ormsgpack" 2>/dev/null; then
    echo "✅ Method 1: Using Python 3.9 user packages"
    python3 -m streamlit run app.py
    exit 0
fi

# Method 2: Try direct streamlit binary
if [ -f "/Users/ellelammba/Library/Python/3.9/bin/streamlit" ]; then
    echo "✅ Method 2: Using direct Streamlit binary"
    /Users/ellelammba/Library/Python/3.9/bin/streamlit run app.py
    exit 0
fi

# Method 3: Try system python3
if python3 -c "import streamlit, requests, ormsgpack" 2>/dev/null; then
    echo "✅ Method 3: Using system Python 3"
    python3 -m streamlit run app.py
    exit 0
fi

# Method 4: Try to install and run
echo "⚠️ Method 4: Installing compatible Streamlit..."
python3 -m pip install --user streamlit==1.39.0 requests ormsgpack

if python3 -c "import streamlit" 2>/dev/null; then
    echo "✅ Installation successful, starting app..."
    python3 -m streamlit run app.py
else
    echo "❌ Failed to install or run Streamlit"
    echo "💡 Manual setup instructions:"
    echo "   1. pip3 install streamlit==1.39.0 requests ormsgpack"
    echo "   2. streamlit run app.py"
    exit 1
fi
