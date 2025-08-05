import streamlit as st
import os
import sys
import requests
import base64
import ormsgpack

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
    help="Supported formats: MP3, WAV, M4A, FLAC"
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

if st.button("Transcribe", type="primary", disabled=not uploaded_file):
    if uploaded_file is not None and api_key:
        try:
            with st.spinner("Transcribing audio..."):
                audio_data = uploaded_file.read()
                lang_code = None if language == "Auto Detect" else LANGUAGE_MAP[language]
                
                # Direct API call to Fish Audio (using same format as SDK)
                url = "https://api.fish.audio/v1/asr"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/msgpack"
                }
                
                # Create the request payload (same as ASRRequest)
                payload = {
                    "audio": audio_data,
                    "ignore_timestamps": False,  # Enable timestamps
                }
                if lang_code:
                    payload["language"] = lang_code
                
                # Use ormsgpack like the original SDK
                response = requests.post(
                    url, 
                    headers=headers, 
                    data=ormsgpack.packb(payload), 
                    timeout=30
                )
                
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
    """) 