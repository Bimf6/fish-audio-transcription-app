# Fish Audio Transcription App

A Streamlit web application for transcribing audio files using the Fish Audio API.

## Features

- 🎤 Upload audio files (MP3, WAV, M4A, FLAC)
- 🌍 Support for multiple languages (Mandarin, English, Cantonese)
- 🔍 Auto-language detection
- 📝 Download transcriptions as text files
- 🔐 Secure API key management

## Prerequisites

- Python 3.10 or higher
- Fish Audio API key

## Installation

1. **Install Python 3.10+** (if not already installed):
   ```bash
   # On macOS with Homebrew
   brew install python@3.10
   ```

2. **Install dependencies**:
   ```bash
   python3.10 -m pip install -r requirements.txt
   ```

3. **Install the Fish Audio SDK**:
   ```bash
   python3.10 -m pip install .
   ```

## Usage

### Option 1: Using the shell script
```bash
./run_app.sh
```

### Option 2: Direct command
```bash
python3.10 -m streamlit run app.py
```

### Option 3: Set environment variable and run
```bash
export FISH_AUDIO_API_KEY="your_api_key_here"
python3.10 -m streamlit run app.py
```

## Configuration

- **API Key**: Enter your Fish Audio API key in the sidebar, or set the `FISH_AUDIO_API_KEY` environment variable
- **Language**: Select the language of your audio file or use "Auto Detect"
- **File Formats**: Supports MP3, WAV, M4A, and FLAC files

## Troubleshooting

- **Python Version Error**: Make sure you're using Python 3.10 or higher
- **Import Errors**: Ensure the Fish Audio SDK is properly installed
- **API Errors**: Verify your API key is correct and has sufficient credits

## Development

To run in development mode:
```bash
python3.10 -m streamlit run app.py --server.port 8501
```

## License

MIT License
