import streamlit as st
import os
import sys

# Add the current directory to Python path for local SDK import
sys.path.insert(0, '.')

try:
    from fish_audio_sdk import Session, ASRRequest
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please check if all dependencies are installed correctly.")
    st.stop()

LANGUAGE_MAP = {
    "Mandarin": "zh",
    "English": "en",
    "Cantonese": "zh-yue"
}

st.set_page_config(
    page_title="Fish Audio Transcription",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Fish Audio Transcription")

# API Key configuration
# Try to get API key from Streamlit secrets first, then environment variables, then default
default_api_key = ""
try:
    if hasattr(st, 'secrets') and "FISH_AUDIO_API_KEY" in st.secrets:
        default_api_key = st.secrets["FISH_AUDIO_API_KEY"]
    else:
        default_api_key = os.getenv("FISH_AUDIO_API_KEY", "")
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    default_api_key = os.getenv("FISH_AUDIO_API_KEY", "")

api_key = st.sidebar.text_input(
    "API Key", 
    value=default_api_key,
    type="password",
    help="Enter your Fish Audio API key. For security, use the Streamlit Cloud secrets manager in production."
)

uploaded_file = st.file_uploader(
    "Upload audio file", 
    type=["mp3", "wav", "m4a", "flac"],
    help="Supported formats: MP3, WAV, M4A, FLAC"
)

language = st.selectbox(
    "Select language", 
    ["Auto Detect", "Mandarin", "English", "Cantonese"],
    help="Choose the language of your audio file or let the system auto-detect"
)

if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""

if st.button("Transcribe", type="primary", disabled=not uploaded_file):
    if uploaded_file is not None and api_key:
        try:
            with st.spinner("Transcribing audio..."):
                audio_data = uploaded_file.read()
                session = Session(api_key)
                lang_code = None if language == "Auto Detect" else LANGUAGE_MAP[language]
                response = session.asr(ASRRequest(audio=audio_data, language=lang_code))
                st.session_state['transcript'] = response.text
                st.success("Transcription completed!")
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            st.session_state['transcript'] = ""
    elif not api_key:
        st.error("Please enter your API key in the sidebar.")
    else:
        st.warning("Please upload an audio file.")

# Display transcript
if st.session_state['transcript']:
    st.subheader("📝 Transcript")
    st.text_area(
        "Transcription Result", 
        st.session_state['transcript'], 
        height=300,
        disabled=True
    )
    
    # Add download button for transcript
    st.download_button(
        label="📥 Download Transcript",
        data=st.session_state['transcript'],
        file_name="transcript.txt",
        mime="text/plain"
    )
else:
    st.info("Upload an audio file and click 'Transcribe' to get started.") 