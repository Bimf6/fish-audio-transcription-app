# 🎵 Fish Audio Transcription App - Running Instructions

## 🚀 Quick Start

### Method 1: Use the Enhanced Runner (Recommended)
```bash
./run_app.sh
```

### Method 2: Use the Simple Runner (Fallback)
```bash
./run_simple.sh
```

### Method 3: Manual Startup
```bash
# Install compatible dependencies
python3 -m pip install --user streamlit==1.39.0 requests ormsgpack

# Run the app
python3 -m streamlit run app.py
```

## 🔧 Troubleshooting

### Problem: Protobuf Import Error
**Error:** `ImportError: cannot import name 'builder' from 'google.protobuf.internal'`

**Solution:** Use the compatible Streamlit version:
```bash
python3 -m pip install --user --force-reinstall streamlit==1.39.0
export PATH="/Users/ellelammba/Library/Python/3.9/bin:$PATH"
python3 -m streamlit run app.py
```

### Problem: Streamlit Not Found
**Solution:** Install in user directory:
```bash
python3 -m pip install --user streamlit requests ormsgpack
```

### Problem: Permission Denied
**Solution:** Make scripts executable:
```bash
chmod +x run_app.sh run_simple.sh
```

## 📱 App Features

Once running, the app provides:
- ✅ **Large File Support**: Up to 100MB files (76MB+ with compression)
- 🔄 **Smart Compression**: Automatic compression for large files
- 🎤 **Speaker Detection**: Identifies different speakers
- ⏰ **Timestamps**: Precise timecodes for each segment
- 📥 **Multiple Exports**: Text, SRT subtitles, JSON
- 🌍 **Multi-language**: Auto-detect or specify language

## 🔑 API Key Required

1. Visit [Fish Audio](https://fish.audio)
2. Sign up/login to your account
3. Generate an API key
4. Enter it in the app sidebar

## 🌐 Access

Once running, open your browser to: **http://localhost:8501**

## 💡 Tips

- Use **run_simple.sh** if **run_app.sh** fails
- For large files (76MB+), enable compression
- MP3 format works best for large files
- Your API key is never stored, only used during the session
