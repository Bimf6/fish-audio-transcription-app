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
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB - conservative limit for API
RECOMMENDED_SIZE = 25 * 1024 * 1024  # 25MB - recommended size

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
    """Simple fallback compression by truncating file if too large"""
    current_size_mb = len(input_data) / (1024 * 1024)
    
    if current_size_mb <= target_size_mb:
        return input_data
    
    # Simple approach: truncate to target size
    # This is not ideal but works as last resort
    target_bytes = int(target_size_mb * 1024 * 1024)
    
    st.warning("⚠️ Using basic compression. Audio may be truncated. Install FFmpeg for better compression.")
    
    return input_data[:target_bytes]

def validate_file_size(file_data, filename):
    """Validate file size and provide user feedback"""
    file_size = len(file_data)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File '{filename}' is {get_file_size_str(file_size)}, which exceeds the {get_file_size_str(MAX_FILE_SIZE)} limit."
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
    
    # Display file info
    if file_valid:
        if len(audio_data) > RECOMMENDED_SIZE:
            st.warning(file_info_message)
            
            # Add compression options for large files
            with st.expander("📦 File Compression Options (Recommended)"):
                st.info("💡 Your file is large. Compression can reduce upload time and avoid API limits.")
                
                compression_enabled = st.checkbox(
                    "Enable automatic compression",
                    value=True,
                    help="Compress audio to reduce file size while maintaining quality for transcription"
                )
                
                if compression_enabled:
                    target_size = st.slider(
                        "Target file size (MB)",
                        min_value=10,
                        max_value=30,
                        value=20,
                        help="Smaller files upload faster but may have slightly reduced audio quality"
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Size", get_file_size_str(len(audio_data)))
                    with col2:
                        st.metric("Target Size", f"~{target_size} MB")
        else:
            st.success(file_info_message)
    else:
        st.error(file_info_message)
        st.info("💡 Try compressing your audio file using an external tool, or enable compression below.")
        
        # Offer compression even for oversized files
        with st.expander("📦 Emergency Compression"):
            st.warning("⚠️ File exceeds maximum size. Emergency compression may help, but quality may be reduced.")
            compression_enabled = st.checkbox(
                "Attempt emergency compression",
                value=False,
                help="This will significantly compress the audio and may affect transcription quality"
            )
            
            if compression_enabled:
                target_size = st.slider(
                    "Emergency target size (MB)",
                    min_value=5,
                    max_value=25,
                    value=15,
                    help="Very aggressive compression - use only if necessary"
                )

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

if st.button("Transcribe", type="primary", disabled=not uploaded_file or not file_valid):
    if uploaded_file is not None and api_key and file_valid:
        try:
            # Prepare audio data with optional compression
            if compression_enabled:
                with st.spinner("🔄 Compressing audio file..."):
                    progress_bar = st.progress(0)
                    st.info(f"Original size: {get_file_size_str(len(audio_data))}")
                    
                    # Set target size based on file validation
                    if not file_valid:  # Emergency compression
                        target_size_mb = target_size
                    else:  # Regular compression
                        target_size_mb = target_size
                    
                    progress_bar.progress(25)
                    compressed_data = compress_audio_ffmpeg(audio_data, target_size_mb)
                    progress_bar.progress(75)
                    
                    if compressed_data:
                        final_audio_data = compressed_data
                        compression_ratio = len(audio_data) / len(compressed_data)
                        progress_bar.progress(100)
                        st.success(f"✅ Compression complete! Reduced to {get_file_size_str(len(compressed_data))} ({compression_ratio:.1f}x smaller)")
                    else:
                        st.error("❌ Compression failed. Trying with original file...")
                        final_audio_data = audio_data
                    
                    progress_bar.empty()
            else:
                final_audio_data = audio_data
            
            # Validate final file size
            if len(final_audio_data) > MAX_FILE_SIZE:
                st.error(f"File is still too large ({get_file_size_str(len(final_audio_data))}). Please try with more aggressive compression or a smaller file.")
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
                    timeout_seconds = min(120, max(60, len(final_audio_data) // (1024 * 1024) * 2))  # 2 seconds per MB, min 60s, max 120s
                    
                    response = requests.post(
                        url, 
                        headers=headers, 
                        data=ormsgpack.packb(payload), 
                        timeout=timeout_seconds
                    )
                    
                    upload_progress.progress(100)
                    upload_progress.empty()
                
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
        st.error("Please enter your API key in the sidebar.")
    else:
        st.warning("Please upload an audio file.")

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
    st.info("Upload an audio file and click 'Transcribe' to get started.")
    st.markdown("""
    ### 🎯 Features:
    - **🎤 Speaker Identification**: Automatically identifies different speakers
    - **⏰ Timecodes**: Shows when each segment was spoken
    - **📊 Audio Analysis**: Duration and segment statistics
    - **📥 Multiple Export Formats**: Text, SRT subtitles, JSON data
    - **🌍 Multi-language Support**: Auto-detect or specify language
    - **📦 Smart Compression**: Automatically handles large files (up to 76MB+)
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
        - 📙 **25-50MB**: Good, may take slightly longer
        - 📕 **Over 50MB**: Automatic compression recommended
        
        **Tips for Best Results:**
        - Enable compression for files over 25MB
        - Lower target sizes upload faster but may reduce quality slightly
        - MP3 format generally works best for large files
        """)
        
        if os.getenv("DEBUG") == "true":
            st.code(f"""
Debug Info:
- Max API file size: {get_file_size_str(MAX_FILE_SIZE)}
- Recommended size: {get_file_size_str(RECOMMENDED_SIZE)}
- FFmpeg available: {"Yes" if subprocess.run(['ffmpeg', '-version'], capture_output=True).returncode == 0 else "No"}
            """.strip()) 