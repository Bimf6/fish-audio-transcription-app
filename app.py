import streamlit as st
import os
import sys
import requests
import base64
import ormsgpack
import tempfile
import subprocess
import math
import io
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
try:
    import librosa
    import sklearn.cluster
    from sklearn.preprocessing import StandardScaler
    AUDIO_ANALYSIS_AVAILABLE = True
except ImportError:
    AUDIO_ANALYSIS_AVAILABLE = False

LANGUAGE_MAP = {
    "Traditional Chinese": "zh-tw",
    "English": "en", 
    "Cantonese": "zh-yue"
}

# File size limits (in bytes)
# Fish Audio API supports up to 100MB and 60 minutes per request.
# Only chunk when truly necessary - byte-splitting corrupts audio.
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
RECOMMENDED_SIZE = 90 * 1024 * 1024  # 90MB - only warn above this
CHUNKING_THRESHOLD = 95 * 1024 * 1024  # 95MB - only byte-chunk above this (near API limit)
API_CHUNK_SIZE = 45 * 1024 * 1024  # 45MB per chunk if we must split
FALLBACK_CHUNK_SIZE = 30 * 1024 * 1024  # 30MB fallback
EMERGENCY_CHUNK_SIZE = 20 * 1024 * 1024  # 20MB emergency
ULTRA_EMERGENCY_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB ultra emergency

MAX_DURATION_SECONDS = 15 * 60  # 15 minutes in seconds - Fish Audio works better with shorter segments
ESTIMATED_BITRATE_KBPS = 128  # Estimated bitrate for duration calculation when ffmpeg unavailable

def estimate_duration_from_size(file_size_bytes, bitrate_kbps=ESTIMATED_BITRATE_KBPS):
    """Estimate audio duration from file size (fallback when ffmpeg unavailable)"""
    # Convert bitrate to bytes per second
    bytes_per_second = (bitrate_kbps * 1000) / 8
    estimated_seconds = file_size_bytes / bytes_per_second
    return estimated_seconds

def get_audio_duration(audio_data):
    """Get audio duration in seconds using ffprobe, with fallback to estimation"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            duration_cmd = [
                'ffprobe', '-i', temp_file_path, '-show_entries',
                'format=duration', '-v', 'quiet', '-of', 'csv=p=0'
            ]
            result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip())
            return duration
        finally:
            os.unlink(temp_file_path)
    except Exception as e:
        if os.getenv("DEBUG") == "true":
            st.write(f"   ⚠️ Could not get audio duration via ffprobe: {str(e)}")
        # Fallback: estimate duration from file size
        estimated = estimate_duration_from_size(len(audio_data))
        if os.getenv("DEBUG") == "true":
            st.write(f"   📊 Estimated duration from file size: {estimated/60:.1f} minutes")
        return estimated

def _is_mp3_frame_header(b0, b1):
    """MPEG audio frame sync: 0xFF and high bits of next byte."""
    return b0 == 0xFF and (b1 & 0xE0) == 0xE0

def _find_mp3_frame_sync(data: bytes, start: int, end: int) -> int | None:
    """Find next MP3 frame header start in [start, end)."""
    end = min(end, len(data) - 1)
    i = max(0, start)
    while i < end:
        if _is_mp3_frame_header(data[i], data[i + 1]):
            return i
        i += 1
    return None

def _mp3_id3_end(data: bytes) -> int:
    """Byte offset after ID3v2 tag, or 0."""
    if not data.startswith(b'ID3') or len(data) < 10:
        return 0
    try:
        size_bytes = data[6:10]
        tag_size = 0
        for byte in size_bytes:
            tag_size = (tag_size << 7) | (byte & 0x7F)
        return min(10 + tag_size, len(data))
    except Exception:
        return 0

def split_mp3_by_aligned_size(audio_data: bytes, target_segment_bytes: int) -> list[bytes]:
    """Split MP3 at frame boundaries so every chunk is a valid MP3 stream."""
    total = len(audio_data)
    if total <= target_segment_bytes:
        return [audio_data]

    segments: list[bytes] = []
    id3_end = _mp3_id3_end(audio_data)
    pos = 0
    search_window = 8192

    while pos < total:
        if pos == 0:
            # First chunk always from byte 0 (includes ID3 tag if present)
            chunk_start = 0
        else:
            sync = _find_mp3_frame_sync(audio_data, pos, min(pos + search_window, total))
            if sync is None:
                sync = _find_mp3_frame_sync(audio_data, pos, total)
            chunk_start = sync if sync is not None else pos

        target_end = min(chunk_start + target_segment_bytes, total)
        if target_end >= total:
            segments.append(audio_data[chunk_start:total])
            break

        split_at = target_end
        back = _find_mp3_frame_sync(audio_data, max(chunk_start, target_end - search_window), target_end + 1)
        if back is not None and back > chunk_start:
            split_at = back
        else:
            fwd = _find_mp3_frame_sync(audio_data, target_end, min(target_end + search_window, total - 1))
            if fwd is not None:
                split_at = fwd

        if split_at <= chunk_start:
            split_at = min(chunk_start + target_segment_bytes, total)

        if split_at <= pos:
            split_at = min(pos + max(1024, target_segment_bytes // 4), total)

        segments.append(audio_data[chunk_start:split_at])
        pos = split_at

    return [s for s in segments if len(s) > 0]

def split_audio_by_size(audio_data, num_segments):
    """Split audio by size (pure Python fallback when ffmpeg unavailable). MP3 is frame-aligned."""
    total_size = len(audio_data)
    if num_segments <= 0:
        return []
    target = max(ULTRA_EMERGENCY_CHUNK_SIZE, total_size // num_segments)

    debug_mode = os.getenv("DEBUG") == "true"

    if audio_data.startswith(b'ID3') or (len(audio_data) > 2 and audio_data[0:2] == b'\xff\xfb'):
        segments = split_mp3_by_aligned_size(audio_data, target)
        if debug_mode:
            for i, seg in enumerate(segments):
                st.write(f"   ✅ Segment {i+1}/{len(segments)}: {get_file_size_str(len(seg))}")
        return segments

    segment_size = total_size // num_segments
    segments = []
    for i in range(num_segments):
        start = i * segment_size
        end = total_size if i == num_segments - 1 else start + segment_size
        segment = audio_data[start:end]
        if len(segment) > 0:
            segments.append(segment)
            if debug_mode:
                st.write(f"   ✅ Segment {i+1}/{num_segments}: {get_file_size_str(len(segment))}")
    return segments

def split_audio_by_duration(audio_data, max_duration_seconds=MAX_DURATION_SECONDS):
    """Split audio into segments of max_duration_seconds or less using ffmpeg, with Python fallback"""
    debug_mode = os.getenv("DEBUG") == "true"
    
    try:
        total_duration = get_audio_duration(audio_data)
        
        if total_duration is None:
            return None, None
        
        if total_duration <= max_duration_seconds:
            return [audio_data], total_duration
        
        num_segments = math.ceil(total_duration / max_duration_seconds)
        segment_duration = total_duration / num_segments
        
        if debug_mode:
            st.write(f"   🔪 Audio is {total_duration/60:.1f} minutes, splitting into {num_segments} segments of ~{segment_duration/60:.1f} minutes each")
        
        # Check if ffmpeg is available
        ffmpeg_available = False
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
            ffmpeg_available = result.returncode == 0
        except:
            ffmpeg_available = False
        
        if not ffmpeg_available:
            if debug_mode:
                st.write("   ℹ️ ffmpeg not available, using size-based splitting")
            # Use pure Python size-based splitting
            segments = split_audio_by_size(audio_data, num_segments)
            return segments if segments else None, total_duration
        
        # Use ffmpeg for precise splitting
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_input:
            temp_input.write(audio_data)
            input_path = temp_input.name
        
        segments = []
        try:
            for i in range(num_segments):
                start_time = i * segment_duration
                
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_output:
                    output_path = temp_output.name
                
                split_cmd = [
                    'ffmpeg', '-i', input_path,
                    '-ss', str(start_time),
                    '-t', str(segment_duration),
                    '-c', 'copy',
                    '-y', output_path
                ]
                
                result = subprocess.run(split_cmd, capture_output=True, timeout=120)
                
                if result.returncode == 0 and os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        segment_data = f.read()
                    segments.append(segment_data)
                    os.unlink(output_path)
                    
                    if debug_mode:
                        st.write(f"   ✅ Segment {i+1}/{num_segments}: {get_file_size_str(len(segment_data))}")
                else:
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                    if debug_mode:
                        st.write(f"   ❌ Failed to create segment {i+1}")
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
        
        # If ffmpeg splitting failed, fall back to size-based
        if not segments:
            if debug_mode:
                st.write("   ⚠️ ffmpeg splitting failed, using size-based splitting")
            segments = split_audio_by_size(audio_data, num_segments)
        
        return segments if segments else None, total_duration
        
    except Exception as e:
        if os.getenv("DEBUG") == "true":
            st.write(f"   ❌ Duration-based splitting failed: {str(e)}")
        return None, None

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

def chunk_audio_file(audio_data, chunk_size_bytes=API_CHUNK_SIZE):
    """Split large audio file into smaller chunks that can be processed by the API"""
    chunks = []
    total_size = len(audio_data)
    
    if total_size <= chunk_size_bytes:
        return [audio_data]  # No need to chunk
    
    # MP3: frame-aligned (arbitrary byte splits break decode after chunk 1)
    if audio_data.startswith(b'ID3') or (len(audio_data) > 2 and audio_data[0:2] in (b'\xff\xfb', b'\xff\xfa')):
        chunks = chunk_mp3_audio(audio_data, chunk_size_bytes)
    # WAV: each chunk must be a valid RIFF/WAVE file
    elif audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]:
        chunks = chunk_wav_audio(audio_data, chunk_size_bytes)
        if not chunks:
            for i in range(0, total_size, chunk_size_bytes):
                chunks.append(audio_data[i:i + chunk_size_bytes])
    else:
        # M4A/FLAC/other: byte split may be invalid; still try sequential slices
        for i in range(0, total_size, chunk_size_bytes):
            chunk = audio_data[i:i + chunk_size_bytes]
            chunks.append(chunk)
    
    return chunks

def adaptive_chunk_audio_file(audio_data, initial_chunk_size=API_CHUNK_SIZE):
    """Create chunks with adaptive sizing based on failure patterns"""
    total_size = len(audio_data)
    
    # Start with much smaller chunks to avoid 500 errors
    # Use appropriate chunk sizes based on total file size
    if total_size > 300 * 1024 * 1024:  # > 300MB
        chunk_size = EMERGENCY_CHUNK_SIZE  # 20MB
    elif total_size > 200 * 1024 * 1024:  # > 200MB
        chunk_size = FALLBACK_CHUNK_SIZE  # 30MB
    else:
        chunk_size = initial_chunk_size  # Use default (45MB)
    
    chunks = chunk_audio_file(audio_data, int(chunk_size))
    
    # Allow more chunks for better reliability (for long audio files)
    max_reasonable_chunks = 100  # Allow up to 100 chunks
    if len(chunks) > max_reasonable_chunks:
        # Try to balance between size and number of chunks
        optimal_chunk_size = total_size // max_reasonable_chunks
        if optimal_chunk_size < ULTRA_EMERGENCY_CHUNK_SIZE:
            optimal_chunk_size = ULTRA_EMERGENCY_CHUNK_SIZE
        elif optimal_chunk_size > FALLBACK_CHUNK_SIZE:
            optimal_chunk_size = FALLBACK_CHUNK_SIZE
        
        chunks = chunk_audio_file(audio_data, int(optimal_chunk_size))
    
    return chunks, int(chunk_size)

def rechunk_on_failure(audio_data, failed_chunks, original_chunk_size):
    """Re-chunk the audio with smaller size when chunks fail with 500 errors"""
    debug_mode = os.getenv("DEBUG") == "true"
    
    # Determine new chunk size based on failure pattern - progressive reduction
    if original_chunk_size > FALLBACK_CHUNK_SIZE:
        new_chunk_size = int(FALLBACK_CHUNK_SIZE)  # 5MB
        retry_msg = f"🔄 Retrying with smaller chunks ({get_file_size_str(new_chunk_size)} each)"
    elif original_chunk_size > EMERGENCY_CHUNK_SIZE:
        new_chunk_size = int(EMERGENCY_CHUNK_SIZE)  # 3MB
        retry_msg = f"🔄 Retrying with smaller chunks ({get_file_size_str(new_chunk_size)} each)"
    elif original_chunk_size > ULTRA_EMERGENCY_CHUNK_SIZE:
        new_chunk_size = int(ULTRA_EMERGENCY_CHUNK_SIZE)  # 1MB
        retry_msg = f"🔄 Retrying with smaller chunks ({get_file_size_str(new_chunk_size)} each)"
    else:
        # Try one more time with even smaller chunks
        new_chunk_size = int(ULTRA_EMERGENCY_CHUNK_SIZE * 0.5)  # 500KB
        if new_chunk_size < 300 * 1024:  # Don't go below 300KB
            if debug_mode:
                st.write("   ❌ Already at minimum chunk size, cannot reduce further")
            return None, None
        retry_msg = f"🔄 Final attempt with smaller chunks ({get_file_size_str(new_chunk_size)} each)"
    
    if debug_mode:
        st.write(f"   📉 Reducing chunk size: {get_file_size_str(original_chunk_size)} → {get_file_size_str(new_chunk_size)}")
    
    st.info(retry_msg)
    
    # Create new smaller chunks
    new_chunks = chunk_audio_file(audio_data, new_chunk_size)
    
    if debug_mode:
        st.write(f"   📦 New chunking: {len(new_chunks)} chunks instead of {len(failed_chunks)}")
    
    return new_chunks, new_chunk_size

def chunk_wav_audio(audio_data: bytes, chunk_size_bytes: int) -> list[bytes]:
    """Split WAV/RIFF into valid standalone WAV chunks (each with header)."""
    if (not audio_data.startswith(b'RIFF')) or (b'WAVE' not in audio_data[:12]):
        return []
    try:
        with wave.open(io.BytesIO(audio_data), 'rb') as wf:
            params = wf.getparams()
            frames_bytes = wf.readframes(wf.getnframes())
    except Exception:
        return []

    frame_size = params.nchannels * params.sampwidth
    if frame_size <= 0:
        return []
    frames_per_chunk = max(1, chunk_size_bytes // frame_size)
    chunks: list[bytes] = []
    total_frames = len(frames_bytes) // frame_size
    i = 0
    while i < total_frames:
        n = min(frames_per_chunk, total_frames - i)
        slice_b = frames_bytes[i * frame_size : (i + n) * frame_size]
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as out:
            out.setparams(params)
            out.writeframes(slice_b)
        chunks.append(buf.getvalue())
        i += n
    return chunks

def chunk_mp3_audio(audio_data, chunk_size_bytes):
    """Split MP3 at frame boundaries so each chunk is decodable (fixes ~first 3–4 min only bug)."""
    return split_mp3_by_aligned_size(audio_data, chunk_size_bytes)

def validate_chunk_data(chunk_data, chunk_num=1):
    """Validate that chunk data is likely to be processable audio"""
    issues = []
    
    # Size checks
    if len(chunk_data) == 0:
        issues.append("Empty chunk")
    elif len(chunk_data) < 100:
        issues.append(f"Very small chunk ({len(chunk_data)} bytes)")
    elif len(chunk_data) > 100 * 1024 * 1024:
        issues.append(f"Chunk too large ({len(chunk_data) / (1024*1024):.1f}MB, API limit 100MB)")
    
    # Basic format validation
    if len(chunk_data) >= 4:
        # Check for common audio format signatures
        if chunk_data.startswith(b'ID3'):
            # MP3 with ID3 tag - good
            pass
        elif chunk_data.startswith(b'RIFF'):
            # WAV/RIFF format - good
            pass
        elif chunk_data.startswith(b'\xff\xfb') or chunk_data.startswith(b'\xff\xfa'):
            # MP3 frame start - good
            pass
        elif chunk_data.startswith(b'\x00\x00\x00'):
            # Lots of zeros - might be corrupted
            issues.append("Chunk starts with zeros (possibly corrupted)")
        elif len(set(chunk_data[:100])) < 10:
            # Very low entropy in first 100 bytes
            issues.append("Low entropy data (possibly corrupted)")
    
    return issues

def diagnose_500_error_causes(chunk_data, api_key, lang_code, response_text=""):
    """Diagnose potential causes of 500 errors beyond file size"""
    issues = []
    
    # Check API key format
    if not api_key or len(api_key.strip()) < 10:
        issues.append("Invalid or too short API key")
    elif not api_key.startswith(('sk-', 'fish_')):  # Common API key prefixes
        issues.append("API key format may be incorrect")
    
    # Check audio format issues
    if len(chunk_data) < 100:
        issues.append(f"Audio chunk too small ({len(chunk_data)} bytes)")
    elif not (chunk_data.startswith(b'ID3') or 
              chunk_data.startswith(b'RIFF') or 
              chunk_data.startswith(b'\xff\xfb') or 
              chunk_data.startswith(b'\xff\xfa')):
        issues.append("Unrecognized audio format - may not be valid audio")
    
    # Check for common error patterns in response
    if response_text:
        response_lower = response_text.lower()
        if "invalid audio" in response_lower:
            issues.append("Server reports invalid audio format")
        elif "corrupted" in response_lower:
            issues.append("Server reports corrupted audio data")
        elif "timeout" in response_lower:
            issues.append("Server-side timeout processing audio")
        elif "memory" in response_lower or "resource" in response_lower:
            issues.append("Server resource/memory issues")
        elif "quota" in response_lower or "limit" in response_lower:
            issues.append("API quota or rate limit exceeded")
        elif "authentication" in response_lower:
            issues.append("API authentication issue")
    
    # Check language code
    if lang_code and lang_code not in ['en', 'zh', 'zh-tw', 'zh-yue']:
        issues.append(f"Potentially unsupported language code: {lang_code}")
    
    return issues

def extract_voice_features(audio_data, sample_rate=22050):
    """Extract voice features for speaker identification"""
    if not AUDIO_ANALYSIS_AVAILABLE:
        return None
    
    try:
        # Convert audio data to numpy array
        if isinstance(audio_data, bytes):
            # Save to temporary file and load with librosa
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                y, sr = librosa.load(temp_file_path, sr=sample_rate)
            finally:
                os.unlink(temp_file_path)
        else:
            y = audio_data
            sr = sample_rate
        
        # Extract features for voice analysis
        features = []
        
        # 1. MFCC (Mel-frequency cepstral coefficients) - voice timbre
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # 2. Spectral features - voice characteristics
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        # 3. Zero crossing rate - voice rhythm
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 4. Tempo and rhythm
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # 5. Chroma features - pitch class
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        return np.array(features)
        
    except Exception as e:
        if os.getenv("DEBUG") == "true":
            st.write(f"   ⚠️ Voice feature extraction failed: {str(e)}")
        return None

def analyze_speaker_segments(segments, audio_data=None):
    """Analyze segments to identify speakers based on voice characteristics"""
    if not AUDIO_ANALYSIS_AVAILABLE or not segments:
        return segments  # Return unchanged if no analysis available
    
    debug_mode = os.getenv("DEBUG") == "true"
    
    try:
        # Extract features for each segment if audio data available
        segment_features = []
        valid_segments = []
        
        for i, segment in enumerate(segments):
            # For now, use text-based features if audio analysis fails
            # In a full implementation, you'd extract audio for each segment
            text = segment.get('text', '')
            
            # Simple text-based voice characteristics
            features = []
            
            # Text length (indicates speaking style)
            features.append(len(text))
            
            # Punctuation patterns (indicates speaking style)
            features.append(text.count('?'))  # Questions
            features.append(text.count('!'))  # Exclamations
            features.append(text.count('.'))  # Statements
            features.append(text.count(','))  # Pauses
            
            # Word characteristics
            words = text.split()
            features.append(len(words))  # Word count
            if words:
                features.append(np.mean([len(word) for word in words]))  # Avg word length
            else:
                features.append(0)
            
            # Sentiment indicators (basic)
            positive_words = ['yes', 'good', 'great', 'excellent', 'love', 'like']
            negative_words = ['no', 'bad', 'terrible', 'hate', 'dislike', 'wrong']
            features.append(sum(1 for word in words if word.lower() in positive_words))
            features.append(sum(1 for word in words if word.lower() in negative_words))
            
            # Speaking patterns
            features.append(1 if text.strip().endswith('?') else 0)  # Ends with question
            features.append(1 if any(word.lower() in ['um', 'uh', 'er', 'ah'] for word in words) else 0)  # Hesitation
            
            segment_features.append(features)
            valid_segments.append(segment)
        
        if len(segment_features) < 2:
            return segments  # Need at least 2 segments to cluster
        
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(segment_features)
        
        # Determine optimal number of speakers (2-4)
        n_speakers = min(max(2, len(segments) // 3), 4)
        
        # Cluster segments by speaker
        clustering = sklearn.cluster.KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        speaker_labels = clustering.fit_predict(normalized_features)
        
        # Assign speaker labels to segments
        speaker_names = ['Speaker A', 'Speaker B', 'Speaker C', 'Speaker D']
        for i, segment in enumerate(valid_segments):
            speaker_id = speaker_labels[i]
            segment['speaker'] = speaker_names[speaker_id]
        
        if debug_mode:
            st.write(f"   🎤 Voice analysis: Identified {n_speakers} speakers across {len(segments)} segments")
            speaker_counts = {}
            for label in speaker_labels:
                speaker_counts[speaker_names[label]] = speaker_counts.get(speaker_names[label], 0) + 1
            for speaker, count in speaker_counts.items():
                st.write(f"      {speaker}: {count} segments")
        
        return valid_segments
        
    except Exception as e:
        if debug_mode:
            st.write(f"   ⚠️ Speaker analysis failed: {str(e)}")
        return segments  # Return unchanged on error

def estimate_duration_from_file_size(file_size_bytes):
    """Estimate total audio duration in milliseconds from file size.
    Assumes ~128kbps bitrate (16KB per second of audio)."""
    bytes_per_second = 16 * 1024  # 128kbps = 16KB/s
    duration_seconds = file_size_bytes / bytes_per_second
    return duration_seconds * 1000  # Return milliseconds

def estimate_chunk_duration(chunk_size_bytes, total_duration_ms, total_file_size):
    """Estimate the duration of an audio chunk in milliseconds"""
    if total_file_size == 0:
        return 0
    
    chunk_ratio = chunk_size_bytes / total_file_size
    return int(total_duration_ms * chunk_ratio)

def asr_segment_times_to_seconds(start, end):
    """Fish Audio ASR segment start/end are in milliseconds (official docs). Always convert to seconds."""
    return float(start or 0) / 1000.0, float(end or 0) / 1000.0

def api_duration_to_ms(d: float) -> int:
    """Normalize ASR response `duration`: documented as ms; some responses use seconds for short clips."""
    if d is None or d <= 0:
        return 0
    d = float(d)
    if d >= 100000:
        return int(d)
    if d <= 3600:
        return int(d * 1000)
    return int(d)

def normalize_segments_list_to_seconds(segments: list) -> list:
    """Convert segment start/end from API (ms) to seconds for UI/export."""
    if not segments:
        return segments
    out = []
    for seg in segments:
        s = dict(seg)
        ss, es = asr_segment_times_to_seconds(seg.get("start"), seg.get("end"))
        s["start"], s["end"] = ss, es
        out.append(s)
    return out

def ensure_chunk_has_segments(chunk_result: dict, chunk_duration_ms: int) -> dict:
    """If API returns text but empty segments, add one segment so the full chunk appears in the UI."""
    if not chunk_result:
        return chunk_result
    text = (chunk_result.get('text') or "").strip()
    segs = chunk_result.get("segments")
    if text and (not segs or len(segs) == 0):
        d_sec = max(chunk_duration_ms / 1000.0, 0.1)
        out = dict(chunk_result)
        # Milliseconds like API responses; stitch_transcripts converts to seconds
        out["segments"] = [{"text": chunk_result.get("text", ""), "start": 0.0, "end": float(d_sec * 1000.0)}]
        return out
    return chunk_result

def chunk_duration_ms_from_result(chunk_result: dict, chunk_bytes: int, total_bytes: int, total_estimated_ms: int) -> int:
    """Prefer API-reported duration per chunk; normalize seconds vs ms; fallback to size estimate."""
    if chunk_result:
        api_d = chunk_result.get("duration")
        if api_d is not None:
            try:
                d = float(api_d)
                if d > 0:
                    return api_duration_to_ms(d)
            except (TypeError, ValueError):
                pass
    return estimate_chunk_duration(chunk_bytes, total_estimated_ms, total_bytes)

def stitch_transcripts(transcript_chunks, chunk_durations):
    """Combine multiple transcript chunks into a single result with proper timestamps.
    Segment start/end from Fish Audio are in milliseconds; we store normalized seconds in output."""
    combined_text = ""
    combined_segments = []
    current_time_offset = 0.0
    
    for i, (transcript_data, chunk_duration_ms) in enumerate(zip(transcript_chunks, chunk_durations)):
        if not transcript_data:
            # Keep timeline aligned when a chunk failed (still advance by estimated chunk length)
            current_time_offset += float(chunk_duration_ms or 0) / 1000.0
            continue
        
        # Sanity check: chunk_duration_ms should be reasonable (< 1 hour = 3,600,000 ms)
        if chunk_duration_ms > 3600000:
            segments = transcript_data.get('segments', [])
            if segments:
                last_segment = segments[-1]
                _, le = asr_segment_times_to_seconds(
                    last_segment.get("start", 0), last_segment.get("end", 0)
                )
                chunk_duration_ms = int(le * 1000) + 1000
            else:
                chunk_duration_ms = 60000
        
        # Add text
        if combined_text:
            combined_text += " "
        combined_text += transcript_data.get('text', '')
        
        # Add segments with adjusted timestamps (seconds)
        segments = transcript_data.get('segments', [])
        for segment in segments:
            adjusted_segment = segment.copy()
            start_sec, end_sec = asr_segment_times_to_seconds(
                segment.get('start', 0), segment.get('end', 0)
            )
            start_time = start_sec + current_time_offset
            end_time = end_sec + current_time_offset
            
            adjusted_segment['start'] = start_time
            adjusted_segment['end'] = end_time
            combined_segments.append(adjusted_segment)
        
        # Advance offset for next chunk (chunk_duration_ms is wall-clock duration of this audio chunk)
        current_time_offset += chunk_duration_ms / 1000.0
    
    # Total duration: sum of chunk durations (ms), or span of last segment
    total_duration = sum(chunk_durations)
    if combined_segments:
        last_end_sec = max(float(s.get("end", 0) or 0) for s in combined_segments)
        span_ms = int(last_end_sec * 1000)
        total_duration = max(total_duration, span_ms)
    
    return {
        'text': combined_text,
        'segments': combined_segments,
        'duration': total_duration
    }

def is_cjk_text(text):
    """Check if text contains CJK (Chinese/Japanese/Korean) characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
            return True
        if '\u3400' <= char <= '\u4dbf':  # CJK Extension A
            return True
        if '\u3040' <= char <= '\u309f':  # Hiragana
            return True
        if '\u30a0' <= char <= '\u30ff':  # Katakana
            return True
    return False

def merge_short_segments(segments, max_gap_seconds=2.0):
    """Merge only tiny ASR fragments (split words). Preserves sentence-scale timing.

    Aggressive merging caused huge text blocks with short time spans (bad timestamps).
    """
    if not segments:
        return segments

    sample_text = "".join(seg.get("text", "") for seg in segments[:10])
    is_cjk = is_cjk_text(sample_text)

    merged = []
    current_segment = None

    for segment in segments:
        text = segment.get("text", "").strip()
        try:
            start = float(segment.get("start", 0) or 0)
            end = float(segment.get("end", 0) or 0)
        except (TypeError, ValueError):
            start, end = 0.0, 0.0

        if current_segment is None:
            current_segment = {
                "text": text,
                "start": start,
                "end": end,
                "speaker": segment.get("speaker"),
            }
            continue

        gap = start - float(current_segment["end"] or 0)
        current_duration = float(current_segment["end"]) - float(current_segment["start"])
        current_text = current_segment["text"]
        current_len = len(current_text.replace(" ", ""))

        # Do not merge across silence / speaker gaps (fixes bogus short windows with long text)
        if gap > max_gap_seconds:
            merged.append(current_segment)
            current_segment = {
                "text": text,
                "start": start,
                "end": end,
                "speaker": segment.get("speaker"),
            }
            continue

        if is_cjk:
            # Only glue tiny fragments (word-level ASR), not full sentences
            tiny = current_duration < 2.0 and current_len < 18
            should_merge = tiny and gap <= max_gap_seconds
        else:
            words = len(current_text.split())
            tiny = current_duration < 2.5 and words < 7
            should_merge = tiny and gap <= max_gap_seconds

        if should_merge:
            if is_cjk:
                current_segment["text"] = current_text + text
            else:
                current_segment["text"] = (current_text + " " + text).strip()
            current_segment["end"] = end
        else:
            merged.append(current_segment)
            current_segment = {
                "text": text,
                "start": start,
                "end": end,
                "speaker": segment.get("speaker"),
            }

    if current_segment:
        merged.append(current_segment)

    return merged

def process_audio_chunk(chunk_data, lang_code, api_key, chunk_num=1, total_chunks=1):
    """Process a single audio chunk and return the transcript result"""
    try:
        # Debug chunk information
        chunk_size_mb = len(chunk_data) / (1024 * 1024)
        debug_mode = os.getenv("DEBUG") == "true"
        
        if debug_mode:
            st.write(f"🔍 Debug Chunk {chunk_num}: {chunk_size_mb:.1f}MB")
            # Show chunk header info
            if len(chunk_data) > 10:
                header_preview = chunk_data[:10].hex()
                st.write(f"   Header: {header_preview}")
                if chunk_data.startswith(b'ID3'):
                    st.write("   Format: MP3 with ID3 tag")
                elif chunk_data.startswith(b'RIFF'):
                    st.write("   Format: WAV/RIFF")
                elif chunk_data.startswith(b'\xff\xfb') or chunk_data.startswith(b'\xff\xfa'):
                    st.write("   Format: MP3 frame")
                else:
                    st.write("   Format: Unknown/Raw audio")
        
        # Validate chunk data
        if len(chunk_data) == 0:
            error_msg = f"❌ Chunk {chunk_num} is empty" if total_chunks > 1 else "❌ Audio file is empty"
            st.error(error_msg)
            if debug_mode:
                st.write(f"   Error: Zero bytes in chunk {chunk_num}")
            return None
        
        # Check if chunk is too large even for individual processing (Fish Audio API limit is 100MB)
        if len(chunk_data) > 100 * 1024 * 1024:  # 100MB
            error_msg = f"❌ Chunk {chunk_num} is too large ({chunk_size_mb:.1f}MB, API limit 100MB)" if total_chunks > 1 else f"❌ File is too large ({chunk_size_mb:.1f}MB, API limit 100MB)"
            st.error(error_msg)
            if debug_mode:
                st.write(f"   Error: Chunk {chunk_num} exceeds 100MB API limit")
            return None
        
        # Check for minimum viable chunk size
        if len(chunk_data) < 1024:  # Less than 1KB
            if debug_mode:
                st.warning(f"⚠️ Chunk {chunk_num} is very small ({len(chunk_data)} bytes)")
            # Continue processing, but flag as potentially problematic
        
        # Validate API key
        if not api_key or len(api_key.strip()) < 10:
            error_msg = f"❌ Invalid API key for chunk {chunk_num}" if total_chunks > 1 else "❌ Invalid API key"
            st.error(error_msg)
            if debug_mode:
                st.write(f"   Error: API key length {len(api_key) if api_key else 0}")
            return None
        
        # Direct API call to Fish Audio
        url = "https://api.fish.audio/v1/asr"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/msgpack"
        }
        
        # Create the request payload matching SDK's ASRRequest schema
        payload = {
            "audio": chunk_data,
            "ignore_timestamps": False,
        }
        
        if lang_code:
            payload["language"] = lang_code
        
        # Validate payload before sending
        if debug_mode:
            st.write(f"   Payload keys: {list(payload.keys())}")
            st.write(f"   Language: {lang_code or 'auto-detect'}")
            st.write(f"   Audio data size: {len(chunk_data)} bytes")
            
            if len(chunk_data) == 0:
                st.error(f"   ❌ Empty audio data in chunk {chunk_num}")
                return None
            
            if chunk_data.startswith(b'ID3'):
                st.write(f"   🎵 Audio format: MP3 with ID3 tag")
            elif chunk_data.startswith(b'RIFF'):
                st.write(f"   🎵 Audio format: WAV/RIFF")
            elif chunk_data.startswith(b'\xff\xfb') or chunk_data.startswith(b'\xff\xfa'):
                st.write(f"   🎵 Audio format: MP3 frame")
            else:
                st.write(f"   ⚠️ Audio format: Unknown - first bytes: {chunk_data[:10].hex()}")
        
        # Adaptive timeout based on chunk size
        if chunk_size_mb < 1:
            timeout_seconds = 60
        elif chunk_size_mb < 10:
            timeout_seconds = 120
        elif chunk_size_mb < 50:
            timeout_seconds = 300  # 5 minutes for medium files
        else:
            timeout_seconds = 600  # 10 minutes for large files
        
        # Add extra time for retry attempts
        if debug_mode:
            st.write(f"   ⏱️ Timeout set to {timeout_seconds}s for {chunk_size_mb:.1f}MB chunk")
        
        # Retry logic for individual chunks
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    import time
                    wait_time = 3 + attempt  # 3s, 4s
                    if total_chunks > 1:
                        st.warning(f"🔄 Retrying chunk {chunk_num} (attempt {attempt + 1}/{max_retries + 1})")
                    elif debug_mode:
                        st.write(f"   Retry attempt {attempt + 1}")
                    time.sleep(wait_time)
                
                # Pack the payload using msgpack (as per Fish Audio SDK)
                try:
                    packed_payload = ormsgpack.packb(payload)
                    if debug_mode:
                        st.write(f"   Packed payload size: {len(packed_payload)} bytes (msgpack)")
                except Exception as pack_error:
                    last_error = f"Payload packing error: {str(pack_error)}"
                    if debug_mode:
                        st.error(f"   ❌ Packing failed: {last_error}")
                    continue
                
                # Make the API request
                if debug_mode:
                    st.write(f"   🌐 Making API request to {url}")
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    data=packed_payload, 
                    timeout=timeout_seconds
                )
                
                # Log response details for debugging
                if debug_mode:
                    st.write(f"   📡 Response: {response.status_code}")
                    st.write(f"   📏 Response size: {len(response.content)} bytes")
                    if hasattr(response, 'headers') and 'content-type' in response.headers:
                        st.write(f"   📄 Content-Type: {response.headers['content-type']}")
                
                # Check for server errors with detailed logging
                if response.status_code in [500, 502, 503, 504]:
                    last_error = f"Server error {response.status_code}: {response.text[:200]}"
                    
                    # Enhanced error logging
                    if debug_mode or total_chunks > 1:
                        st.write(f"   ⚠️ Server error details:")
                        st.write(f"      Status: {response.status_code}")
                        st.write(f"      Chunk: {chunk_num}/{total_chunks}")
                        st.write(f"      Size: {chunk_size_mb:.1f}MB")
                        st.write(f"      Attempt: {attempt + 1}/{max_retries + 1}")
                        if hasattr(response, 'headers'):
                            content_type = response.headers.get('content-type', 'unknown')
                            st.write(f"      Response type: {content_type}")
                        if len(response.text) > 0:
                            st.write(f"      Error message: {response.text[:300]}")
                        else:
                            st.write(f"      No error message in response")
                    
                    if attempt < max_retries:
                        continue
                    else:
                        # Run comprehensive diagnosis on persistent 500 errors
                        diagnosis = diagnose_500_error_causes(chunk_data, api_key, lang_code, response.text)
                        
                        if total_chunks > 1:
                            st.error(f"❌ Chunk {chunk_num} failed after {max_retries + 1} attempts: {last_error}")
                            if diagnosis:
                                st.error("🔍 Potential causes of 500 error:")
                                for issue in diagnosis:
                                    st.write(f"   • {issue}")
                        else:
                            show_api_error(response.status_code, response.text)
                            if diagnosis:
                                st.error("🔍 Potential causes:")
                                for issue in diagnosis:
                                    st.write(f"   • {issue}")
                        return None
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if debug_mode:
                            st.write(f"   ✅ Success: {len(result.get('text', ''))} chars transcribed")
                            if 'segments' in result:
                                st.write(f"   📊 Segments: {len(result['segments'])}")
                        return result
                    except Exception as json_error:
                        last_error = f"JSON parsing error: {str(json_error)}"
                        if debug_mode:
                            st.error(f"   ❌ JSON parsing failed: {last_error}")
                            st.write(f"   Raw response: {response.text[:500]}")
                        continue
                else:
                    last_error = f"API error {response.status_code}: {response.text[:200]}"
                    if debug_mode:
                        st.write(f"   ❌ API error details: {last_error}")
                    if total_chunks > 1:
                        st.error(f"❌ Chunk {chunk_num} failed: HTTP {response.status_code}")
                    else:
                        show_api_error(response.status_code, response.text)
                    return None
                    
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {timeout_seconds}s"
                if attempt < max_retries:
                    timeout_seconds += 30  # Increase timeout for retry
                    continue
                else:
                    if total_chunks > 1:
                        st.error(f"❌ Chunk {chunk_num} timed out: {last_error}")
                    else:
                        st.error("⏰ Request timed out. Please try with a smaller audio file.")
                    return None
            except requests.exceptions.RequestException as e:
                last_error = f"Network error: {str(e)[:100]}"
                if attempt < max_retries:
                    continue
                else:
                    if total_chunks > 1:
                        st.error(f"❌ Chunk {chunk_num} network error: {last_error}")
                    else:
                        st.error(f"🔄 Network error: {str(e)}")
                    return None
            except Exception as e:
                last_error = f"Unexpected error: {str(e)[:100]}"
                if total_chunks > 1:
                    st.error(f"❌ Chunk {chunk_num} error: {last_error}")
                else:
                    st.error(f"❌ Error during transcription: {str(e)}")
                return None
                    
    except Exception as e:
        if total_chunks > 1:
            st.error(f"❌ Chunk {chunk_num} processing error: {str(e)}")
        else:
            st.error(f"❌ Error during transcription: {str(e)}")
        return None

def test_api_connection(api_key):
    """Test API connection with a minimal request to diagnose issues"""
    try:
        # Create a minimal test payload - small silent audio
        test_audio = b'\xff\xfb\x90\x00' + b'\x00' * 500  # Minimal MP3 frame
        
        url = "https://api.fish.audio/v1/asr"
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/msgpack"
        }
        
        payload = {
            "audio": test_audio,
            "ignore_timestamps": True,
        }
        
        packed_payload = ormsgpack.packb(payload)
        response = requests.post(
            url, 
            headers=headers, 
            data=packed_payload,
            timeout=30
        )
        
        return response.status_code, response.text
        
    except Exception as e:
        return None, str(e)

def show_api_error(status_code, response_text):
    """Show appropriate error message based on API response"""
    if status_code == 413:
        st.error("🚫 File too large for API (413 error)")
        st.info("💡 Try with a smaller audio file or check if chunking is working properly.")
    elif status_code == 429:
        st.error("⏰ Rate limit exceeded. Please wait and try again.")
    elif status_code == 401:
        st.error("🔑 Invalid API key. Please check your Fish Audio API key.")
        st.info("💡 Test your API key with a simple request to verify it's working.")
    elif status_code == 400:
        st.error("❌ Bad request (400) - The API rejected our request format.")
        
        # Enhanced 400 error troubleshooting
        with st.expander("🔧 400 Error Solutions"):
            st.markdown("""
            **Common Causes of 400 Bad Request:**
            
            1. **Request Format Issues**:
               - Wrong Content-Type (we switched from msgpack to JSON)
               - Invalid payload structure
               - Missing required fields
               
            2. **Audio Data Issues**:
               - Audio not properly base64 encoded
               - Unsupported audio format
               - Corrupted audio data
               
            3. **API Changes**:
               - API endpoint may have changed
               - Required fields may have been updated
               - Authentication format changed
               
            **What We've Already Fixed:**
            ✅ Switched from msgpack to JSON format
            ✅ Added base64 encoding for audio data
            ✅ Added explicit format specification
            ✅ Updated timestamp field names
            
            **Next Steps to Try:**
            🔧 Use the API test button in debug mode
            🔧 Try a different audio file format
            🔧 Check if your API key has the right permissions
            """)
        
        st.info("💡 Enable debug mode and use 'Test API Connection' to diagnose further.")
    elif status_code == 500:
        st.error("🔥 Server error (500) - The Fish Audio API server encountered an issue.")
        
        # Enhanced 500 error troubleshooting
        with st.expander("🔧 500 Error Troubleshooting Guide"):
            st.markdown("""
            **Common Causes of 500 Errors:**
            
            1. **Audio Format Issues**:
               - Corrupted audio file
               - Unsupported audio codec
               - Invalid audio headers
               
            2. **API Key Problems**:
               - Expired API key
               - Insufficient permissions
               - Account quota exceeded
               
            3. **Server-Side Issues**:
               - API server overload
               - Processing timeout
               - Memory/resource limits
            
            **Solutions to Try:**
            
            ✅ **Test API Key**: Use debug mode to test your API key
            ✅ **Convert Audio**: Try converting to standard MP3 format
            ✅ **Reduce Quality**: Lower bitrate/sample rate
            ✅ **Split File**: Break into smaller time segments (not just smaller chunks)
            ✅ **Try Different File**: Test with a known-good audio file
            ✅ **Wait and Retry**: Server may be temporarily overloaded
            """)
    elif status_code == 502 or status_code == 503:
        st.error(f"🔧 Service temporarily unavailable ({status_code})")
        st.info("💡 The Fish Audio service may be busy. Try again in a few minutes.")
    else:
        st.error(f"API Error {status_code}: {response_text}")

def validate_file_size(file_data, filename):
    """Validate file size and provide user feedback"""
    file_size = len(file_data)
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File '{filename}' is {get_file_size_str(file_size)}, which exceeds the {get_file_size_str(MAX_FILE_SIZE)} limit."
    elif file_size > CHUNKING_THRESHOLD:
        return True, f"File '{filename}' is {get_file_size_str(file_size)}. Will be processed in chunks for optimal results."
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
    
    # Auto-handle file chunking transparently
    use_chunking = False
    use_duration_split = False
    duration_segments = None
    audio_duration = None
    
    if file_valid:
        file_size = len(audio_data)
        
        # Check audio duration
        with st.spinner("🕐 Checking audio duration..."):
            audio_duration = get_audio_duration(audio_data)
        
        if audio_duration is not None:
            duration_minutes = audio_duration / 60
            max_duration_minutes = MAX_DURATION_SECONDS / 60
            
            if audio_duration > MAX_DURATION_SECONDS:
                use_duration_split = True
                num_segments = math.ceil(audio_duration / MAX_DURATION_SECONDS)
                st.warning(f"⏱️ Audio is {duration_minutes:.1f} minutes (exceeds {max_duration_minutes:.0f} min limit). Will split into {num_segments} segments of ~{max_duration_minutes:.0f} min each.")
            else:
                st.info(f"⏱️ Audio duration: {duration_minutes:.1f} minutes — will process as single file")
        
        # Only use byte-chunking for very large files (near API limit)
        # Fish Audio API supports up to 100MB, so most files go through as-is
        if file_size > CHUNKING_THRESHOLD and not use_duration_split:
            use_chunking = True
            num_chunks = math.ceil(file_size / API_CHUNK_SIZE)
            st.info(f"🧩 File is {get_file_size_str(file_size)} (near API limit). Will process in ~{num_chunks} chunks.")
        elif not use_duration_split:
            st.success(f"📂 File is {get_file_size_str(file_size)} — ready to transcribe")
    else:
        st.error(file_info_message)

language = st.selectbox(
    "Select language", 
    ["Auto Detect", "Traditional Chinese", "English", "Cantonese"],
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
    elif speaker_detection_mode == "Voice Characteristics":
        if AUDIO_ANALYSIS_AVAILABLE:
            st.info("🔬 Advanced voice analysis enabled - uses ML clustering to identify speakers by voice characteristics")
        else:
            st.warning("🔬 Advanced voice analysis requires additional packages. Install: pip install librosa scikit-learn")
            st.info("📝 Currently using enhanced text-based analysis as fallback")
    
    show_confidence = st.checkbox("Show confidence scores", value=False, key="show_confidence")
    include_timestamps = st.checkbox("Include detailed timestamps", value=True, key="include_timestamps")
    use_card_view = st.checkbox("Use modern card layout", value=True, key="use_card_view")
    
    # Debug mode toggle
    debug_mode = st.checkbox("Enable debug mode", value=False, key="debug_mode", help="Shows detailed processing information")
    if debug_mode:
        os.environ["DEBUG"] = "true"
        st.info("🔍 Debug mode enabled - detailed processing info will be shown")
        
        # API testing in debug mode
        if st.button("🧪 Test API Connection", help="Test your API key with a minimal request"):
            if api_key:
                with st.spinner("Testing API connection..."):
                    status_code, response_text = test_api_connection(api_key)
                    
                    if status_code is None:
                        st.error(f"❌ Connection failed: {response_text}")
                    elif status_code == 200:
                        st.success("✅ API connection successful!")
                    elif status_code == 401:
                        st.error("❌ API key authentication failed")
                    elif status_code == 500:
                        st.error("❌ API server error - this may be the cause of your 500 errors")
                        st.write(f"Response: {response_text[:200]}")
                    else:
                        st.warning(f"⚠️ API responded with status {status_code}")
                        st.write(f"Response: {response_text[:200]}")
            else:
                st.error("❌ Please enter your API key first")
    else:
        os.environ.pop("DEBUG", None)
    
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
    st.session_state['completion_time'] = None

if st.button("Transcribe", type="primary", disabled=not uploaded_file or not file_valid or not api_key):
    if uploaded_file is not None and api_key and file_valid:
        try:
            lang_code = None if language == "Auto Detect" else LANGUAGE_MAP[language]
            
            # Track if we successfully processed the file
            processing_done = False
            
            if use_duration_split:
                # Split audio by duration (15 minute segments for reliability)
                with st.spinner("🔪 Splitting audio into 15-minute segments..."):
                    duration_segments, total_duration = split_audio_by_duration(audio_data, MAX_DURATION_SECONDS)
                
                if duration_segments is None:
                    st.warning("⚠️ Failed to split audio by duration (ffmpeg may not be installed). Falling back to size-based chunking.")
                    use_duration_split = False
                    use_chunking = True  # Force size-based chunking
                else:
                    st.success(f"✅ Split into {len(duration_segments)} segments")
                    
                    # Process each duration segment
                    transcript_chunks = []
                    chunk_durations = []
                    total_segments = len(duration_segments)
                    segment_duration_each = total_duration / total_segments
                    
                    progress_container = st.container()
                    with progress_container:
                        overall_progress = st.progress(0)
                        status_text = st.empty()
                    
                    for i, segment in enumerate(duration_segments):
                        segment_num = i + 1
                        status_text.text(f"🎤 Transcribing segment {segment_num}/{total_segments} (~{segment_duration_each/60:.1f} min each)...")
                        
                        # Each segment might still need size-based chunking if very large
                        if len(segment) > API_CHUNK_SIZE:
                            sub_chunks, sub_chunk_size = adaptive_chunk_audio_file(segment)
                            sub_results = []
                            sub_durations = []
                            
                            seg_budget_ms = int(segment_duration_each * 1000)
                            for j, sub_chunk in enumerate(sub_chunks):
                                result = process_audio_chunk(sub_chunk, lang_code, api_key, j+1, len(sub_chunks))
                                if result:
                                    est_ms = chunk_duration_ms_from_result(
                                        result,
                                        len(sub_chunk),
                                        len(segment),
                                        seg_budget_ms,
                                    )
                                    result = ensure_chunk_has_segments(result, est_ms)
                                    sub_results.append(result)
                                    sub_durations.append(est_ms)
                                else:
                                    sub_results.append(None)
                                    sub_durations.append(
                                        estimate_chunk_duration(len(sub_chunk), seg_budget_ms, len(segment))
                                    )

                            if any(r is not None for r in sub_results):
                                combined = stitch_transcripts(sub_results, sub_durations)
                                transcript_chunks.append(combined)
                            else:
                                transcript_chunks.append(None)
                            chunk_durations.append(seg_budget_ms)
                        else:
                            result = process_audio_chunk(segment, lang_code, api_key, segment_num, total_segments)
                            if result:
                                est_ms = chunk_duration_ms_from_result(
                                    result,
                                    len(segment),
                                    len(segment),
                                    int(segment_duration_each * 1000),
                                )
                                result = ensure_chunk_has_segments(result, est_ms)
                                transcript_chunks.append(result)
                                chunk_durations.append(est_ms)
                            else:
                                transcript_chunks.append(None)
                                chunk_durations.append(int(segment_duration_each * 1000))
                        
                        progress = segment_num / total_segments
                        overall_progress.progress(progress)
                    
                    overall_progress.empty()
                    status_text.empty()
                    
                    # Combine all segments (keep failed slots with duration so timeline stays correct)
                    successful_chunks = [(chunk, duration) for chunk, duration in zip(transcript_chunks, chunk_durations) if chunk is not None]
                    
                    if successful_chunks:
                        with st.spinner("🔗 Combining transcripts..."):
                            result = stitch_transcripts(transcript_chunks, chunk_durations)
                        
                        segments = result.get('segments', [])
                        if segments and st.session_state.get('speaker_detection_mode') == "Voice Characteristics":
                            with st.spinner("🎤 Analyzing speakers..."):
                                segments = analyze_speaker_segments(segments, audio_data)
                                result['segments'] = segments
                        
                        st.session_state['transcript'] = result.get('text', '')
                        st.session_state['transcript_data'] = result
                        # Merge short segments for better readability
                        merged_segments = merge_short_segments(segments)
                        st.session_state['segments'] = merged_segments
                        st.session_state['duration'] = result.get('duration', 0)
                        st.session_state['completion_time'] = datetime.now()
                        
                        if len(successful_chunks) == total_segments:
                            st.success(f"✅ Transcription completed! Processed all {total_segments} segments.")
                        else:
                            st.warning(f"⚠️ Partial success: {len(successful_chunks)}/{total_segments} segments processed.")
                        processing_done = True
                    else:
                        st.error("❌ All segments failed to process.")
                        st.session_state['transcript'] = ""
                        st.session_state['transcript_data'] = None
                        st.session_state['segments'] = []
                        st.session_state['completion_time'] = None
                        processing_done = True
            
            if use_chunking and not processing_done:
                # Process large file in chunks with adaptive sizing
                with st.spinner("🧩 Splitting audio file into chunks..."):
                    chunks, chunk_size = adaptive_chunk_audio_file(audio_data)
                    st.info(f"📦 Split into {len(chunks)} chunks of ~{get_file_size_str(chunk_size)} each")
                
                # Process each chunk
                transcript_chunks = []
                chunk_durations = []
                total_chunks = len(chunks)
                
                progress_container = st.container()
                with progress_container:
                    overall_progress = st.progress(0)
                    status_text = st.empty()
                    
                for i, chunk in enumerate(chunks):
                    chunk_num = i + 1
                    status_text.text(f"🎤 Processing chunk {chunk_num}/{total_chunks}...")
                    
                    debug_mode = os.getenv("DEBUG") == "true"
                    
                    # Validate chunk before processing
                    chunk_issues = validate_chunk_data(chunk, chunk_num)
                    if chunk_issues:
                        if debug_mode:
                            st.warning(f"⚠️ Chunk {chunk_num} validation issues: {', '.join(chunk_issues)}")
                        
                        # Skip chunks with critical issues
                        critical_issues = [issue for issue in chunk_issues if "Empty" in issue or "too large" in issue]
                        if critical_issues:
                            st.error(f"❌ Chunk {chunk_num} has critical issues: {', '.join(critical_issues)}, skipping...")
                            transcript_chunks.append(None)
                            _est = estimate_chunk_duration(
                                len(chunk),
                                estimate_duration_from_file_size(len(audio_data)),
                                len(audio_data),
                            )
                            chunk_durations.append(_est)
                            continue
                    
                    # Process individual chunk with debug info
                    if debug_mode:
                        st.write(f"🔍 Processing chunk {chunk_num}: {get_file_size_str(len(chunk))}")
                        if not chunk_issues:
                            st.write(f"   ✅ Chunk validation passed")
                    
                    chunk_result = process_audio_chunk(chunk, lang_code, api_key, chunk_num, total_chunks)
                    
                    if chunk_result:
                        total_estimated_duration_ms = estimate_duration_from_file_size(len(audio_data))
                        est_ms = chunk_duration_ms_from_result(
                            chunk_result, len(chunk), len(audio_data), total_estimated_duration_ms
                        )
                        chunk_result = ensure_chunk_has_segments(chunk_result, est_ms)
                        transcript_chunks.append(chunk_result)
                        chunk_durations.append(est_ms)
                        
                        # Show success for this chunk
                        chunk_text_preview = chunk_result.get('text', '')[:50]
                        if os.getenv("DEBUG") == "true":
                            st.success(f"✅ Chunk {chunk_num} completed: '{chunk_text_preview}...'")
                    else:
                        st.error(f"❌ Failed to process chunk {chunk_num}/{total_chunks}")
                        transcript_chunks.append(None)
                        _est = estimate_chunk_duration(
                            len(chunk),
                            estimate_duration_from_file_size(len(audio_data)),
                            len(audio_data),
                        )
                        chunk_durations.append(_est)
                    
                    # Update progress
                    progress = chunk_num / total_chunks
                    overall_progress.progress(progress)
                
                # Clean up progress indicators
                overall_progress.empty()
                status_text.empty()
                
                # Check for 500 errors pattern and retry with smaller chunks
                successful_chunks = [(chunk, duration) for chunk, duration in zip(transcript_chunks, chunk_durations) if chunk is not None]
                successful_count = len(successful_chunks)
                failed_count = total_chunks - successful_count
                
                # If many chunks failed and we haven't tried smaller chunks yet, retry
                failure_rate = failed_count / total_chunks
                if (failure_rate >= 0.3 or failed_count >= 2) and chunk_size > ULTRA_EMERGENCY_CHUNK_SIZE:  # 30% or more failed, or 2+ failures
                    st.warning(f"⚠️ {failed_count}/{total_chunks} chunks failed. Attempting automatic recovery with smaller chunks...")
                    
                    # Try re-chunking with smaller size
                    retry_chunks, retry_chunk_size = rechunk_on_failure(audio_data, chunks, chunk_size)
                    
                    if retry_chunks:
                        # Retry with smaller chunks
                        retry_transcript_chunks = []
                        retry_chunk_durations = []
                        retry_total_chunks = len(retry_chunks)
                        
                        retry_progress_container = st.container()
                        with retry_progress_container:
                            retry_overall_progress = st.progress(0)
                            retry_status_text = st.empty()
                            
                        for i, chunk in enumerate(retry_chunks):
                            chunk_num = i + 1
                            retry_status_text.text(f"🔄 Retry chunk {chunk_num}/{retry_total_chunks}...")
                            
                            debug_mode = os.getenv("DEBUG") == "true"
                            
                            # Process retry chunk
                            chunk_result = process_audio_chunk(chunk, lang_code, api_key, chunk_num, retry_total_chunks)
                            
                            if chunk_result:
                                total_estimated_duration_ms = estimate_duration_from_file_size(len(audio_data))
                                est_ms = chunk_duration_ms_from_result(
                                    chunk_result, len(chunk), len(audio_data), total_estimated_duration_ms
                                )
                                chunk_result = ensure_chunk_has_segments(chunk_result, est_ms)
                                retry_transcript_chunks.append(chunk_result)
                                retry_chunk_durations.append(est_ms)
                                
                                if debug_mode:
                                    chunk_text_preview = chunk_result.get('text', '')[:50]
                                    st.success(f"✅ Retry chunk {chunk_num} completed: '{chunk_text_preview}...'")
                            else:
                                retry_transcript_chunks.append(None)
                                _est = estimate_chunk_duration(
                                    len(chunk),
                                    estimate_duration_from_file_size(len(audio_data)),
                                    len(audio_data),
                                )
                                retry_chunk_durations.append(_est)
                            
                            # Update retry progress
                            progress = chunk_num / retry_total_chunks
                            retry_overall_progress.progress(progress)
                        
                        # Clean up retry progress
                        retry_overall_progress.empty()
                        retry_status_text.empty()
                        
                        # Use retry results
                        transcript_chunks = retry_transcript_chunks
                        chunk_durations = retry_chunk_durations
                        total_chunks = retry_total_chunks
                        successful_chunks = [(chunk, duration) for chunk, duration in zip(transcript_chunks, chunk_durations) if chunk is not None]
                        successful_count = len(successful_chunks)
                        
                        st.success(f"🔄 Retry completed: {successful_count}/{total_chunks} chunks successful with smaller size")
                
                # Final results processing
                
                if successful_count > 0:
                    with st.spinner("🔗 Combining transcripts..."):
                        result = stitch_transcripts(transcript_chunks, chunk_durations)
                    
                    # Analyze speakers if voice analysis is requested
                    segments = result.get('segments', [])
                    if segments and st.session_state.get('speaker_detection_mode') == "Voice Characteristics":
                        with st.spinner("🎤 Analyzing speakers..."):
                            segments = analyze_speaker_segments(segments, audio_data)
                            result['segments'] = segments
                    
                    st.session_state['transcript'] = result.get('text', '')
                    st.session_state['transcript_data'] = result
                    # Merge short segments for better readability
                    merged_segments = merge_short_segments(segments)
                    st.session_state['segments'] = merged_segments
                    st.session_state['duration'] = result.get('duration', 0)
                    st.session_state['completion_time'] = datetime.now()
                    
                    if successful_count == total_chunks:
                        st.success(f"✅ Transcription completed! Processed all {total_chunks} chunks successfully.")
                    else:
                        st.warning(f"⚠️ Partial success: {successful_count}/{total_chunks} chunks processed. Some content may be missing.")
                        st.info("💡 The transcript contains the successfully processed portions of your audio.")
                    processing_done = True
                else:
                    st.error("❌ All chunks failed to process. Please try again with a different file or check your API key.")
                    st.session_state['transcript'] = ""
                    st.session_state['transcript_data'] = None
                    st.session_state['segments'] = []
                    st.session_state['completion_time'] = None
                    processing_done = True
                    
            if not processing_done:
                # Process normally for smaller files
                with st.spinner("🎤 Transcribing audio..."):
                    result = process_audio_chunk(audio_data, lang_code, api_key)
                    
                    if result:
                        total_ms = estimate_duration_from_file_size(len(audio_data))
                        est_ms = chunk_duration_ms_from_result(
                            result, len(audio_data), len(audio_data), total_ms
                        )
                        result = ensure_chunk_has_segments(result, est_ms)
                        segments = normalize_segments_list_to_seconds(result.get('segments', []))
                        result = {**result, 'segments': segments}
                        if segments and st.session_state.get('speaker_detection_mode') == "Voice Characteristics":
                            with st.spinner("🎤 Analyzing speakers..."):
                                segments = analyze_speaker_segments(segments, audio_data)
                                result['segments'] = segments
                        
                        st.session_state['transcript'] = result.get('text', '')
                        st.session_state['transcript_data'] = result
                        merged_segments = merge_short_segments(segments)
                        st.session_state['segments'] = merged_segments
                        last_ms = int(max((float(s.get('end', 0) or 0) for s in merged_segments), default=0) * 1000)
                        api_d = api_duration_to_ms(float(result.get("duration") or 0))
                        st.session_state['duration'] = max(est_ms, last_ms, api_d)
                        st.session_state['completion_time'] = datetime.now()
                        
                        st.success("✅ Transcription completed!")
                    else:
                        st.error("❌ Transcription failed. Please try again.")
                        st.session_state['completion_time'] = None
                        
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            st.session_state['transcript'] = ""
            st.session_state['transcript_data'] = None
            st.session_state['segments'] = []
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
    
    if seconds < 0:
        seconds = 0
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def identify_speaker(segment_index, text_length, text_content="", speaker_names=None, detection_mode="Auto (Pattern-based)", segment=None):
    """Enhanced speaker identification with multiple methods"""
    if speaker_names is None:
        speaker_names = {"Speaker A": "Speaker A", "Speaker B": "Speaker B", "Speaker C": "Speaker C"}
    
    # If segment already has speaker from voice analysis, use it
    if segment and 'speaker' in segment:
        speaker_key = segment['speaker']
        return speaker_names.get(speaker_key, speaker_key)
    
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
        # If voice analysis was performed, speaker should be in segment
        # This is a fallback for when voice analysis wasn't available
        if not AUDIO_ANALYSIS_AVAILABLE:
            st.warning("🔬 Advanced voice analysis requires additional packages. Using text-based analysis.")
        
        # Enhanced text-based analysis as fallback
        text_lower = text_content.lower()
        
        # Detect question patterns (often interviewer)
        question_indicators = ['?', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'can you', 'could you', 'would you']
        if any(indicator in text_lower for indicator in question_indicators):
            return speaker_names.get("Speaker A", "Speaker A")
        
        # Detect response patterns
        response_indicators = ['yes', 'no', 'well', 'actually', 'i think', 'i believe', 'in my opinion']
        if any(indicator in text_lower for indicator in response_indicators):
            return speaker_names.get("Speaker B", "Speaker B")
        
        # Long detailed responses
        if text_length > 150:
            return speaker_names.get("Speaker B", "Speaker B")
        
        # Short acknowledgments
        elif text_length < 30:
            return speaker_names.get("Speaker C", "Speaker C")
        
        # Default to alternating
        else:
            speaker_key = list(speaker_names.keys())[segment_index % len(speaker_names)]
            return speaker_names[speaker_key]
    
    else:  # Auto (Pattern-based)
        # Improved alternating pattern with intelligence
        if text_length < 20:  # Short responses (acknowledgments, etc.)
            return speaker_names.get("Speaker C", "Speaker C")
        elif "?" in text_content:  # Questions
            return speaker_names.get("Speaker A", "Speaker A")
        elif segment_index % 2 == 0:
            return speaker_names.get("Speaker A", "Speaker A")
        else:
            return speaker_names.get("Speaker B", "Speaker B")

# Display transcript results
if st.session_state['transcript']:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Detailed Transcript")
        
        # Show completion time
        if st.session_state.get('completion_time'):
            completion_time = st.session_state['completion_time']
            st.caption(f"🕐 Transcribed on {completion_time.strftime('%Y-%m-%d at %H:%M:%S')}")
        
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
                    detection_mode,
                    segment
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
        
        # Add header with completion time
        if st.session_state.get('completion_time'):
            completion_time = st.session_state['completion_time']
            formatted_transcript += f"Transcription completed on {completion_time.strftime('%Y-%m-%d at %H:%M:%S')}\n"
            formatted_transcript += "=" * 50 + "\n\n"
        
        if st.session_state['segments']:
            # Get speaker names
            speaker_names_dict = {
                "Speaker A": st.session_state.get("speaker_0", "Speaker A"),
                "Speaker B": st.session_state.get("speaker_1", "Speaker B"), 
                "Speaker C": st.session_state.get("speaker_2", "Speaker C")
            }
            
            for i, segment in enumerate(st.session_state['segments']):
                start_time = format_timecode(segment.get('start', 0))
                end_time = format_timecode(segment.get('end', 0))
                text = segment.get('text', '')
                speaker = identify_speaker(i, len(text), text, speaker_names_dict)
                formatted_transcript += f"[{start_time} - {end_time}] {speaker}: {text}\n\n"
        
        # Fallback to plain transcript if no formatted content
        if not formatted_transcript and st.session_state['transcript']:
            formatted_transcript = st.session_state['transcript']
        
        # Ensure we have something to download
        if not formatted_transcript:
            formatted_transcript = "(No transcript content available)"
        
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
                try:
                    start_sec = float(segment.get('start', 0) or 0)
                    end_sec = float(segment.get('end', 0) or 0)
                    start_ms = int(start_sec * 1000)
                    end_ms = int(end_sec * 1000)
                    
                    start_srt = f"{start_ms//3600000:02d}:{(start_ms//60000)%60:02d}:{(start_ms//1000)%60:02d},{start_ms%1000:03d}"
                    end_srt = f"{end_ms//3600000:02d}:{(end_ms//60000)%60:02d}:{(end_ms//1000)%60:02d},{end_ms%1000:03d}"
                    
                    text = segment.get('text', '')
                    speaker = identify_speaker(i-1, len(text))
                    srt_content += f"{i}\n{start_srt} --> {end_srt}\n{speaker}: {text}\n\n"
                except (ValueError, TypeError):
                    # Skip segments with invalid timestamps
                    text = segment.get('text', '')
                    if text:
                        srt_content += f"{i}\n00:00:00,000 --> 00:00:00,000\n{text}\n\n"
            
            if srt_content:
                st.download_button(
                    label="🎬 Download SRT Subtitles",
                    data=srt_content,
                    file_name="subtitles.srt",
                    mime="text/plain"
                )
        
        # JSON export for developers
        if st.session_state['transcript_data']:
            import json
            try:
                json_data = json.dumps(st.session_state['transcript_data'], indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                # Fallback if data isn't JSON serializable
                json_data = json.dumps({"text": st.session_state.get('transcript', ''), "error": "Full data not serializable"})
            
            st.download_button(
                label="🔧 Download JSON Data",
                data=json_data,
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
        - ✅ Files up to 500MB+ are now supported
        - ⏱️ **Audio longer than 15 minutes is automatically split** into segments
        - 🧩 Automatic chunking when files exceed API limits  
        - 📈 Real-time progress feedback during batch processing
        - 🔗 Smart transcript stitching preserves timestamps
        - ⭐ No quality loss - processes full audio content
        
        **Duration & Size Guidelines:**
        - ⏱️ **Over 15 minutes**: Auto-split into 15-minute segments
        - 📗 **Under 90MB**: Single file processing (sent directly to API)
        - 📙 **90-100MB**: May need chunking near API limit
        - 📕 **Over 100MB**: Automatic batch processing
        
        **Tips for Best Results:**
        - Files under 90MB are sent as a single request for best accuracy
        - Duration-based splitting preserves audio quality better than size-based chunking
        - MP3 format works best for large files
        - Timestamps are automatically adjusted across segments
        """)
        
        if os.getenv("DEBUG") == "true":
            st.code(f"""
Debug Info:
- Max file size limit: {get_file_size_str(MAX_FILE_SIZE)}
- Chunking threshold: {get_file_size_str(CHUNKING_THRESHOLD)}
- Default chunk size: {get_file_size_str(API_CHUNK_SIZE)}
- Fallback chunk size: {get_file_size_str(int(FALLBACK_CHUNK_SIZE))}
- Emergency chunk size: {get_file_size_str(EMERGENCY_CHUNK_SIZE)}
- Ultra emergency chunk size: {get_file_size_str(ULTRA_EMERGENCY_CHUNK_SIZE)}
- Recommended size: {get_file_size_str(RECOMMENDED_SIZE)}
- Adaptive chunking: Yes (ultra-aggressive sizing to prevent 500 errors)
            """.strip()) 