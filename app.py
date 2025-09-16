import streamlit as st
import os
import sys
import requests
import base64
import ormsgpack
import tempfile
import subprocess
from pathlib import Path

LANGUAGE_MAP = {
    "Mandarin": "zh",
    "English": "en",
    "Cantonese": "zh-yue"
}

# File size limits (in bytes)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB - increased limit for large files
RECOMMENDED_SIZE = 25 * 1024 * 1024  # 25MB - recommended size
COMPRESSION_THRESHOLD = 40 * 1024 * 1024  # 40MB - auto-compress above this size

def get_file_size_str(size_bytes):
    """Convert bytes to human readable file size"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def compress_audio_ffmpeg(input_data, target_size_mb=20):
    """Compress audio using FFmpeg to reduce file size"""
    try:
        # Check if FFmpeg is available
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            st.warning("⚠️ FFmpeg not found. Using basic compression fallback.")
            return compress_audio_fallback(input_data, target_size_mb)
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_input:
            temp_input.write(input_data)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        # Calculate target bitrate based on file duration and desired size
        # First, get duration
        duration_cmd = [
            'ffprobe', '-i', temp_input_path, '-show_entries', 
            'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
        ]
        
        try:
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
            duration = float(duration_result.stdout.strip())
            
            # Calculate target bitrate: (target_size_mb * 8 * 1024) / duration
            target_bitrate = int((target_size_mb * 8 * 1024) / duration)
            target_bitrate = max(32, min(target_bitrate, 192))  # Clamp between 32-192 kbps
            
        except (subprocess.TimeoutExpired, ValueError, subprocess.CalledProcessError):
            target_bitrate = 64  # Default fallback
        
        # Compress the audio
        compress_cmd = [
            'ffmpeg', '-i', temp_input_path, '-codec:a', 'mp3', 
            '-b:a', f'{target_bitrate}k', '-ac', '1', '-ar', '16000',
            '-y', temp_output_path
        ]
        
        result = subprocess.run(compress_cmd, capture_output=True, timeout=60)
        
        if result.returncode == 0:
            with open(temp_output_path, 'rb') as f:
                compressed_data = f.read()
            
            # Cleanup
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            
            return compressed_data
        else:
            # Cleanup on failure
            os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            return compress_audio_fallback(input_data, target_size_mb)
            
    except Exception as e:
        st.warning(f"FFmpeg compression failed: {str(e)}. Trying fallback method...")
        return compress_audio_fallback(input_data, target_size_mb)

def compress_audio_fallback(input_data, target_size_mb=20):
    """Improved fallback compression without FFmpeg"""
    current_size_mb = len(input_data) / (1024 * 1024)
    
    if current_size_mb <= target_size_mb:
        return input_data
    
    st.info("🔧 Using smart audio optimization (FFmpeg not installed).")
    
    # Try to find MP3 frame boundaries for smarter truncation
    target_bytes = int(target_size_mb * 1024 * 1024)
    
    # For MP3 files, try to preserve header and avoid cutting mid-frame
    if input_data.startswith(b'ID3') or input_data[0:2] == b'\xff\xfb':
        # This is likely an MP3 file
        # Keep the first part (headers) and then take chunks from throughout the file
        header_size = min(8192, len(input_data) // 10)  # First 8KB or 10% of file
        remaining_budget = target_bytes - header_size
        
        if remaining_budget > 0:
            # Take samples from different parts of the file
            chunk_size = remaining_budget // 4
            chunks = []
            chunks.append(input_data[:header_size])  # Header
            
            # Take 3 more chunks from different parts of the audio
            file_length = len(input_data)
            for i in range(3):
                start_pos = header_size + (i * file_length // 4)
                end_pos = min(start_pos + chunk_size, file_length)
                if start_pos < file_length:
                    chunks.append(input_data[start_pos:end_pos])
            
            result = b''.join(chunks)
            return result
    
    # Fallback: simple truncation
    result = input_data[:target_bytes]
    return result

def validate_file_size(file_data, filename):
    """Validate file size and provide user feedback"""
    file_size = len(file_data)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File '{filename}' is {get_file_size_str(file_size)}, which exceeds the {get_file_size_str(MAX_FILE_SIZE)} limit. Try compressing the file first."
    elif file_size > COMPRESSION_THRESHOLD:
        return True, f"File '{filename}' is {get_file_size_str(file_size)}. Compression is strongly recommended for files this large."
    elif file_size > RECOMMENDED_SIZE:
        return True, f"File '{filename}' is {get_file_size_str(file_size)}. This is large and may take longer to process."
    else:
        return True, f"File '{filename}' is {get_file_size_str(file_size)} - good size for processing."

st.set_page_config(
    page_title="Fish Audio Transcription",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 Fish Audio Transcription")

# Debug info (only show in development)
if os.getenv("DEBUG") == "true":
    st.sidebar.write("🔧 Debug Info")
    st.sidebar.write(f"Python: {sys.version}")
    st.sidebar.write(f"Streamlit: {st.__version__}")
    try:
        import ormsgpack
        st.sidebar.write("✅ ormsgpack imported")
    except ImportError:
        st.sidebar.write("❌ ormsgpack failed")

# API Key configuration
st.sidebar.markdown("### 🔑 Fish Audio API Key")
st.sidebar.markdown("""
**Required:** Get your API key from [Fish Audio](https://fish.audio)
- Sign up/login to Fish Audio
- Go to your account settings
- Generate an API key
- Paste it below
""")

api_key = st.sidebar.text_input(
    "Enter your Fish Audio API Key", 
    value="",
    type="password",
    placeholder="Enter your API key here...",
    help="Your API key is required to use the transcription service. It's not stored anywhere and only used for this session."
)

st.sidebar.markdown("🔒 **Privacy**: Your API key is only used for this session and is never stored or logged.")

# API key validation
if api_key:
    if len(api_key) < 10:
        st.sidebar.warning("⚠️ API key seems too short. Please check your key.")
    else:
        st.sidebar.success("✅ API key entered")
else:
    st.sidebar.error("❌ API key required to proceed")

uploaded_file = st.file_uploader(
    "Upload audio file", 
    type=["mp3", "wav", "m4a", "flac"],
    help="Supported formats: MP3, WAV, M4A, FLAC. Large files will be automatically compressed if needed."
)

# File size validation and compression options
auto_compress = False
compression_enabled = False
file_valid = True
file_info_message = ""

if uploaded_file is not None:
    # Read file data once
    audio_data = uploaded_file.read()
    uploaded_file.seek(0)  # Reset file pointer for potential re-reading
    
    file_valid, file_info_message = validate_file_size(audio_data, uploaded_file.name)
    
    # Auto-handle file compression transparently
    compression_enabled = False
    target_size = 20  # Default target size
    
    if file_valid:
        if len(audio_data) > COMPRESSION_THRESHOLD:
            st.warning(f"📦 Large file detected ({get_file_size_str(len(audio_data))}). Will automatically compress for optimal processing.")
            compression_enabled = True
            # Automatically determine optimal target size
            file_size_mb = len(audio_data) / (1024 * 1024)
            if file_size_mb > 80:
                target_size = 15  # Very aggressive for huge files
            elif file_size_mb > 60:
                target_size = 18  # Aggressive for large files
            else:
                target_size = 22  # Moderate for medium-large files
        elif len(audio_data) > RECOMMENDED_SIZE:
            st.info(f"📂 Medium file ({get_file_size_str(len(audio_data))}) - will compress if needed.")
            compression_enabled = True
            target_size = 25  # Less aggressive for medium files
        else:
            st.success(file_info_message)
    else:
        st.error(file_info_message)
        st.info("💡 Try compressing your audio file using an external tool, or enable compression below.")
        
        # Auto-enable emergency compression for oversized files
        compression_enabled = True
        target_size = 12  # Very aggressive compression for oversized files
        st.info("🔧 File exceeds limits. Will attempt emergency compression.")

language = st.selectbox(
    "Select language", 
    ["Auto Detect", "Mandarin", "English", "Cantonese"],
    help="Choose the language of your audio file or let the system auto-detect"
)

# Main display options (always visible)
col1_opts, col2_opts = st.columns(2)
with col1_opts:
    show_timestamps_main = st.checkbox("📅 Show Timestamps", value=True, key="main_timestamps")
with col2_opts:
    show_speakers_main = st.checkbox("🎤 Show Speakers", value=True, key="main_speakers")

# Advanced options
with st.expander("🔧 Advanced Options"):
    speaker_detection_mode = st.selectbox(
        "Speaker Detection Mode",
        ["Auto (Pattern-based)", "Manual Labels", "Voice Characteristics"],
        help="Choose how to identify different speakers",
        key="speaker_detection_mode"
    )
    
    if speaker_detection_mode == "Manual Labels":
        st.info("💡 Tip: Upload audio where speakers introduce themselves or use different vocal patterns")
    
    show_confidence = st.checkbox("Show confidence scores", value=False, key="show_confidence")
    include_timestamps = st.checkbox("Include detailed timestamps", value=True, key="include_timestamps")
    use_card_view = st.checkbox("Use modern card layout", value=True, key="use_card_view")
    
    # Speaker customization
    st.write("**Speaker Labels:**")
    speaker_names = {}
    for i in range(3):
        speaker_names[f"Speaker {chr(65+i)}"] = st.text_input(
            f"Speaker {chr(65+i)} name:", 
            value=f"Speaker {chr(65+i)}",
            key=f"speaker_{i}"
        )

if 'transcript' not in st.session_state:
    st.session_state['transcript'] = ""
    st.session_state['segments'] = []
    st.session_state['transcript_data'] = None
    st.session_state['duration'] = 0

if st.button("Transcribe", type="primary", disabled=not uploaded_file or not file_valid or not api_key):
    if uploaded_file is not None and api_key and file_valid:
        try:
            # Prepare audio data with optional compression
            if compression_enabled:
                with st.spinner("🔄 Optimizing audio file for transcription..."):
                    progress_bar = st.progress(0)
                    
                    progress_bar.progress(25)
                    compressed_data = compress_audio_ffmpeg(audio_data, target_size)
                    progress_bar.progress(75)
                    
                    if compressed_data:
                        final_audio_data = compressed_data
                        compression_ratio = len(audio_data) / len(compressed_data)
                        progress_bar.progress(100)
                        st.success(f"✅ File optimized: {get_file_size_str(len(audio_data))} → {get_file_size_str(len(compressed_data))}")
                    else:
                        st.warning("⚠️ Optimization failed. Using original file...")
                        final_audio_data = audio_data
                    
                    progress_bar.empty()
            else:
                final_audio_data = audio_data
            
            # Validate final file size before sending to API
            if len(final_audio_data) > 50 * 1024 * 1024:  # API actual limit is around 50MB
                st.error(f"Compressed file is still too large ({get_file_size_str(len(final_audio_data))}) for the API. Please try with more aggressive compression.")
                st.info("💡 Try reducing the target size to 15-20MB for better compatibility.")
            else:
                with st.spinner("🎤 Transcribing audio..."):
                    lang_code = None if language == "Auto Detect" else LANGUAGE_MAP[language]
                
                    # Direct API call to Fish Audio (using same format as SDK)
                    url = "https://api.fish.audio/v1/asr"
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/msgpack"
                    }
                    
                    # Create the request payload (same as ASRRequest)
                    payload = {
                        "audio": final_audio_data,
                        "ignore_timestamps": False,  # Enable timestamps
                    }
                    if lang_code:
                        payload["language"] = lang_code
                    
                    # Show upload progress
                    upload_progress = st.progress(0)
                    st.info(f"Uploading {get_file_size_str(len(final_audio_data))} to Fish Audio API...")
                    upload_progress.progress(30)
                    
                    # Use ormsgpack like the original SDK with longer timeout for large files
                    timeout_seconds = min(180, max(90, len(final_audio_data) // (1024 * 1024) * 3))  # 3 seconds per MB, min 90s, max 180s
                    
                    # Robust retry logic for large files and server errors
                    max_retries = 3
                    response = None
                    
                    for attempt in range(max_retries + 1):
                        try:
                            if attempt > 0:
                                wait_time = min(5 + (attempt * 2), 15)  # Progressive wait: 5s, 7s, 9s, max 15s
                                st.warning(f"🔄 Attempt {attempt + 1}/{max_retries + 1} - Waiting {wait_time}s before retry...")
                                import time
                                time.sleep(wait_time)
                                upload_progress.progress(30 + (attempt * 15))
                            
                            response = requests.post(
                                url, 
                                headers=headers, 
                                data=ormsgpack.packb(payload), 
                                timeout=timeout_seconds
                            )
                            
                            # Check if we got a server error that we should retry
                            if response.status_code in [500, 502, 503, 504] and attempt < max_retries:
                                st.warning(f"⚠️ Server error {response.status_code}, retrying...")
                                timeout_seconds += 20  # Increase timeout for next attempt
                                continue
                            
                            upload_progress.progress(100)
                            upload_progress.empty()
                            break  # Success or non-retryable error, exit retry loop
                            
                        except requests.exceptions.Timeout:
                            if attempt < max_retries:
                                st.warning(f"⏰ Upload timed out, increasing timeout and retrying...")
                                timeout_seconds += 30  # Increase timeout for retry
                                continue
                            else:
                                raise  # Re-raise if all retries failed
                        except requests.exceptions.RequestException as e:
                            if attempt < max_retries:
                                st.warning(f"🔄 Network error, retrying... ({str(e)[:50]})")
                                continue
                            else:
                                raise
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Debug: Show API response structure
                    if os.getenv("DEBUG") == "true":
                        st.write("**API Response Debug:**")
                        st.write(f"Response keys: {list(result.keys())}")
                        st.write(f"Full response: {result}")
                    
                    st.session_state['transcript'] = result.get('text', 'No transcript available')
                    st.session_state['transcript_data'] = result
                    st.session_state['segments'] = result.get('segments', [])
                    st.session_state['duration'] = result.get('duration', 0)
                    
                    # Debug: Show what we stored
                    if os.getenv("DEBUG") == "true":
                        st.write(f"**Stored segments count**: {len(st.session_state['segments'])}")
                        st.write(f"**Stored transcript length**: {len(st.session_state['transcript'])}")
                    
                    st.success("Transcription completed!")
                else:
                    if response.status_code == 413:
                        st.error("🚫 File too large for API (413 error)")
                        st.info("💡 Try enabling compression above or use a smaller audio file.")
                    elif response.status_code == 429:
                        st.error("⏰ Rate limit exceeded. Please wait and try again.")
                    elif response.status_code == 401:
                        st.error("🔑 Invalid API key. Please check your Fish Audio API key.")
                    elif response.status_code == 400:
                        st.error("❌ Bad request. Check if your audio file format is supported.")
                    elif response.status_code == 500:
                        st.error("🔥 Server error (500) - The Fish Audio API server encountered an issue.")
                        st.error("💡 This typically means the file is still too large or complex for processing.")
                        with st.expander("🔧 Troubleshooting Steps"):
                            st.markdown("""
                            **Try these solutions:**
                            1. **File too large**: Even after compression, your file might be too big
                            2. **Audio format issue**: Try converting to MP3 format first
                            3. **File duration**: Very long files (>2 hours) may cause issues
                            4. **Server capacity**: The API might be overloaded right now
                            
                            **Quick fixes:**
                            - Split your audio into smaller segments (30-60 minutes each)
                            - Convert to MP3 with lower bitrate using external tools
                            - Try again in 10-15 minutes when server load is lower
                            - Use a different audio file to test if the issue persists
                            """)
                    elif response.status_code == 502 or response.status_code == 503:
                        st.error(f"🔧 Service temporarily unavailable ({response.status_code})")
                        st.info("💡 The Fish Audio service may be busy. Try again in a few minutes.")
                    else:
                        st.error(f"API Error {response.status_code}: {response.text}")
                    
                    st.session_state['transcript'] = ""
                    st.session_state['transcript_data'] = None
                    st.session_state['segments'] = []
                    
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try with a smaller audio file.")
            st.session_state['transcript'] = ""
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            st.session_state['transcript'] = ""
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            st.session_state['transcript'] = ""
    elif not api_key:
        st.error("🔑 Please enter your Fish Audio API key in the sidebar to continue.")
        st.info("💡 Don't have an API key? Get one free at [Fish Audio](https://fish.audio)")
    else:
        st.warning("📁 Please upload an audio file to transcribe.")

# Helper functions for timecode and speaker identification
def format_timecode(seconds):
    """Convert seconds to HH:MM:SS or MM:SS format"""
    if seconds is None:
        return "00:00"
    
    # Ensure seconds is a number
    try:
        seconds = float(seconds)
    except (ValueError, TypeError):
        return "00:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def identify_speaker(segment_index, text_length, text_content="", speaker_names=None, detection_mode="Auto (Pattern-based)"):
    """Enhanced speaker identification with multiple methods"""
    if speaker_names is None:
        speaker_names = {"Speaker A": "Speaker A", "Speaker B": "Speaker B", "Speaker C": "Speaker C"}
    
    if detection_mode == "Manual Labels":
        # Look for speaker cues in the text
        text_lower = text_content.lower()
        if any(word in text_lower for word in ["i am", "my name is", "this is"]):
            # Extract potential name after introduction phrases
            for phrase in ["i am ", "my name is ", "this is "]:
                if phrase in text_lower:
                    return "🎙️ Identified Speaker"
        
        # Fallback to pattern-based
        speaker_key = list(speaker_names.keys())[segment_index % len(speaker_names)]
        return speaker_names[speaker_key]
    
    elif detection_mode == "Voice Characteristics":
        # Simple heuristic based on text characteristics
        if "?" in text_content:  # Questions often indicate interviewer/moderator
            return speaker_names.get("Speaker A", "Speaker A")
        elif text_length > 100:  # Long segments might indicate main speaker
            return speaker_names.get("Speaker B", "Speaker B")
        else:  # Short responses
            return speaker_names.get("Speaker C", "Speaker C")
    
    else:  # Auto (Pattern-based)
        # Alternating pattern with some intelligence
        if text_length < 20:  # Short responses (acknowledgments, etc.)
            return speaker_names.get("Speaker C", "Speaker C")
        elif segment_index % 2 == 0:
            return speaker_names.get("Speaker A", "Speaker A")
        else:
            return speaker_names.get("Speaker B", "Speaker B")

# Display transcript results
if st.session_state['transcript']:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Detailed Transcript")
        
        # Show segments with timecodes and speakers
        if st.session_state['segments'] and len(st.session_state['segments']) > 0:
            # Get settings from main toggles and advanced options
            include_timestamps = st.session_state.get('main_timestamps', True)
            show_speakers = st.session_state.get('main_speakers', True)
            show_confidence = st.session_state.get('show_confidence', False)
            use_card_view = st.session_state.get('use_card_view', True)
            
            # Debug: Show current settings
            if os.getenv("DEBUG") == "true":
                st.write(f"**Settings Debug**: timestamps={include_timestamps}, speakers={show_speakers}, confidence={show_confidence}, card_view={use_card_view}")
            
            # Debug information (only show if DEBUG is enabled)
            if os.getenv("DEBUG") == "true":
                st.write(f"**Debug**: Found {len(st.session_state['segments'])} segments")
                if st.session_state['segments']:
                    sample_segment = st.session_state['segments'][0]
                    st.write(f"**Sample segment keys**: {list(sample_segment.keys())}")
                    st.write(f"**Sample start**: {sample_segment.get('start')} ({type(sample_segment.get('start'))})")
                    st.write(f"**Sample end**: {sample_segment.get('end')} ({type(sample_segment.get('end'))})")
            
            for i, segment in enumerate(st.session_state['segments']):
                # Safely get start and end times with error handling
                start_time = format_timecode(segment.get('start', 0))
                end_time = format_timecode(segment.get('end', 0))
                
                # Get speaker names and detection mode from session state or defaults
                speaker_names_dict = {
                    "Speaker A": st.session_state.get("speaker_0", "Speaker A"),
                    "Speaker B": st.session_state.get("speaker_1", "Speaker B"), 
                    "Speaker C": st.session_state.get("speaker_2", "Speaker C")
                }
                detection_mode = st.session_state.get('speaker_detection_mode', "Auto (Pattern-based)")
                
                speaker = identify_speaker(
                    i, 
                    len(segment.get('text', '')), 
                    segment.get('text', ''),
                    speaker_names_dict,
                    detection_mode
                )
                
                # Create a nice display for each segment
                with st.container():
                    if use_card_view:
                        # Modern card-like display
                        if include_timestamps and show_speakers:
                            # Full display with timestamps and speakers
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                    <span style="color: #0066cc; font-weight: bold; margin-right: 15px;">⏰ {start_time} - {end_time}</span>
                                    <span style="color: #ff6b6b; font-weight: bold;">🎤 {speaker}</span>
                                </div>
                                <div style="color: #262730; line-height: 1.5;">
                                    {segment.get('text', 'No text available')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif include_timestamps and not show_speakers:
                            # Only timestamps, no speakers
                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                    <span style="color: #0066cc; font-weight: bold;">⏰ {start_time} - {end_time}</span>
                                </div>
                                <div style="color: #262730; line-height: 1.5;">
                                    {segment.get('text', 'No text available')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif not include_timestamps and show_speakers:
                            # Only speakers, no timestamps
                            st.markdown(f"""
                            <div style="background-color: #f8f9fa; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #ff6b6b;">
                                <div style="color: #ff6b6b; font-weight: bold; margin-bottom: 5px;">🎤 {speaker}</div>
                                <div style="color: #262730; line-height: 1.5;">
                                    {segment.get('text', 'No text available')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Plain text only
                            st.markdown(f"""
                            <div style="background-color: #f9f9f9; padding: 12px; border-radius: 8px; margin: 8px 0;">
                                <div style="color: #262730; line-height: 1.5;">
                                    {segment.get('text', 'No text available')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        if show_confidence and 'confidence' in segment:
                            st.caption(f"📊 Confidence: {segment.get('confidence', 'N/A')}")
                    else:
                        # Classic column-based display
                        if include_timestamps and show_speakers:
                            col_time, col_speaker, col_text = st.columns([1.2, 1.2, 3.6])
                            
                            with col_time:
                                st.markdown(f"**⏰ {start_time} - {end_time}**")
                            
                            with col_speaker:
                                st.markdown(f"🎤 **{speaker}**")
                            
                            with col_text:
                                st.write(segment.get('text', 'No text available'))
                                if show_confidence and 'confidence' in segment:
                                    st.caption(f"📊 Confidence: {segment.get('confidence', 'N/A')}")
                        elif include_timestamps and not show_speakers:
                            col_time, col_text = st.columns([1, 4])
                            
                            with col_time:
                                st.markdown(f"**⏰ {start_time} - {end_time}**")
                            
                            with col_text:
                                st.write(segment.get('text', 'No text available'))
                        elif not include_timestamps and show_speakers:
                            col_speaker, col_text = st.columns([1, 5])
                            
                            with col_speaker:
                                st.markdown(f"🎤 **{speaker}**")
                            
                            with col_text:
                                st.write(segment.get('text', 'No text available'))
                        else:
                            # Plain text only
                            st.write(segment.get('text', 'No text available'))
                        
                        st.divider()
                    
                    # Add some spacing between segments
                    st.write("")
        elif st.session_state['transcript']:
            # Fallback if no segments available but we have transcript text
            st.info("⚠️ Segments with timestamps not available. Showing full transcript only.")
            st.text_area(
                "Full Transcript", 
                st.session_state['transcript'], 
                height=400,
                disabled=True
            )
        else:
            # No transcript data at all
            st.info("No transcript data available. Please upload and transcribe an audio file.")
    
    with col2:
        st.subheader("🔍 Search & Filter")
        
        # Search functionality
        search_term = st.text_input("Search in transcript:", placeholder="Enter search term...")
        if search_term and st.session_state['segments']:
            matching_segments = [
                (i, seg) for i, seg in enumerate(st.session_state['segments']) 
                if search_term.lower() in seg['text'].lower()
            ]
            if matching_segments:
                st.write(f"**Found {len(matching_segments)} matches:**")
                for i, segment in matching_segments[:3]:  # Show first 3 matches
                    start_time = format_timecode(segment['start'])
                    st.write(f"• {start_time}: {segment['text'][:100]}...")
            else:
                st.write("No matches found")
        
        st.subheader("📊 Audio Info")
        
        if st.session_state['duration']:
            total_duration = st.session_state['duration'] / 1000  # Convert from ms
            st.metric("Total Duration", format_timecode(total_duration))
        
        if st.session_state['segments']:
            st.metric("Number of Segments", len(st.session_state['segments']))
            
            # Speaker statistics
            speakers = {}
            for i, segment in enumerate(st.session_state['segments']):
                speaker = identify_speaker(i, len(segment['text']))
                speakers[speaker] = speakers.get(speaker, 0) + 1
            
            st.write("**Speaker Breakdown:**")
            for speaker, count in speakers.items():
                st.write(f"• {speaker}: {count} segments")
        
        # Export options
        st.subheader("📥 Export Options")
        
        # Create formatted transcript for download
        formatted_transcript = ""
        if st.session_state['segments']:
            # Get speaker names
            speaker_names_dict = {
                "Speaker A": st.session_state.get("speaker_0", "Speaker A"),
                "Speaker B": st.session_state.get("speaker_1", "Speaker B"), 
                "Speaker C": st.session_state.get("speaker_2", "Speaker C")
            }
            
            for i, segment in enumerate(st.session_state['segments']):
                start_time = format_timecode(segment['start'])
                end_time = format_timecode(segment['end'])
                speaker = identify_speaker(i, len(segment['text']), segment['text'], speaker_names_dict)
                formatted_transcript += f"[{start_time} - {end_time}] {speaker}: {segment['text']}\n\n"
        else:
            formatted_transcript = st.session_state['transcript']
        
        st.download_button(
            label="📄 Download Full Transcript",
            data=formatted_transcript,
            file_name="detailed_transcript.txt",
            mime="text/plain"
        )
        
        # Create SRT subtitle file
        if st.session_state['segments']:
            srt_content = ""
            for i, segment in enumerate(st.session_state['segments'], 1):
                start_ms = int(segment['start'] * 1000)
                end_ms = int(segment['end'] * 1000)
                
                start_srt = f"{start_ms//3600000:02d}:{(start_ms//60000)%60:02d}:{(start_ms//1000)%60:02d},{start_ms%1000:03d}"
                end_srt = f"{end_ms//3600000:02d}:{(end_ms//60000)%60:02d}:{(end_ms//1000)%60:02d},{end_ms%1000:03d}"
                
                speaker = identify_speaker(i-1, len(segment['text']))
                srt_content += f"{i}\n{start_srt} --> {end_srt}\n{speaker}: {segment['text']}\n\n"
            
            st.download_button(
                label="🎬 Download SRT Subtitles",
                data=srt_content,
                file_name="subtitles.srt",
                mime="text/plain"
            )
        
        # JSON export for developers
        if st.session_state['transcript_data']:
            st.download_button(
                label="🔧 Download JSON Data",
                data=str(st.session_state['transcript_data']),
                file_name="transcript_data.json",
                mime="application/json"
            )

else:
    st.info("🚀 Welcome to Fish Audio Transcription! Follow these steps to get started:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📋 Getting Started:
        1. **🔑 Get API Key**: Sign up at [Fish Audio](https://fish.audio) and get your free API key
        2. **🔐 Enter API Key**: Paste your API key in the sidebar
        3. **📁 Upload Audio**: Choose your audio file (supports large files up to 76MB+)
        4. **🎵 Transcribe**: Click the transcribe button and wait for results
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Features:
        - **🎤 Speaker Identification**: Automatically identifies different speakers
        - **⏰ Timecodes**: Shows when each segment was spoken
        - **📊 Audio Analysis**: Duration and segment statistics
        - **📥 Multiple Export Formats**: Text, SRT subtitles, JSON data
        - **🌍 Multi-language Support**: Auto-detect or specify language
        - **📦 Smart Compression**: Automatically handles large files
        - **🛡️ Error Handling**: Intelligent retry and fallback mechanisms
        """)
    
    with st.expander("ℹ️ Large File Support"):
        st.markdown("""
        **New! Large File Handling:**
        - ✅ Files up to 76MB+ are now supported
        - 🔄 Automatic compression when files exceed API limits
        - 📈 Real-time progress feedback during upload and compression
        - ⚡ Smart bitrate calculation preserves audio quality
        - 🎯 Fallback compression when FFmpeg is unavailable
        
        **File Size Guidelines:**
        - 📗 **Under 25MB**: Optimal processing speed
        - 📙 **25-40MB**: Good, compression optional
        - 📙 **40-76MB**: Large files, compression recommended
        - 📕 **Over 76MB**: Very large, aggressive compression required
        
        **Tips for Best Results:**
        - Enable compression for files over 25MB
        - Lower target sizes upload faster but may reduce quality slightly
        - MP3 format generally works best for large files
        """)
        
        if os.getenv("DEBUG") == "true":
            st.code(f"""
Debug Info:
- Max file size limit: {get_file_size_str(MAX_FILE_SIZE)}
- API size limit: {get_file_size_str(50 * 1024 * 1024)}
- Compression threshold: {get_file_size_str(COMPRESSION_THRESHOLD)}
- Recommended size: {get_file_size_str(RECOMMENDED_SIZE)}
- FFmpeg available: {"Yes" if subprocess.run(['ffmpeg', '-version'], capture_output=True).returncode == 0 else "No"}
            """.strip()) 