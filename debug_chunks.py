#!/usr/bin/env python3
"""
Debug script to test chunk processing without the full Streamlit app
Usage: python debug_chunks.py <audio_file> [api_key]
"""

import sys
import os
import requests
import ormsgpack
from pathlib import Path

# Add the current directory to path so we can import from app.py
sys.path.append('.')
from app import (chunk_audio_file, adaptive_chunk_audio_file, validate_chunk_data, 
                 API_CHUNK_SIZE, FALLBACK_CHUNK_SIZE, EMERGENCY_CHUNK_SIZE, get_file_size_str)

def test_chunk_processing(file_path, api_key=None):
    """Test chunk processing on a real audio file"""
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"🧪 Testing chunk processing for: {file_path}")
    
    # Read the file
    with open(file_path, 'rb') as f:
        audio_data = f.read()
    
    print(f"📁 Original file size: {get_file_size_str(len(audio_data))}")
    
    # Test adaptive chunking
    print(f"\n🧩 Testing adaptive chunking...")
    try:
        chunks, chunk_size = adaptive_chunk_audio_file(audio_data)
        print(f"✅ Adaptive chunking: {len(chunks)} chunks of {get_file_size_str(chunk_size)} each")
        
        # Also test standard chunking for comparison
        print(f"\n📊 Comparison with standard chunking ({get_file_size_str(API_CHUNK_SIZE)}):")
        standard_chunks = chunk_audio_file(audio_data, API_CHUNK_SIZE)
        print(f"   Standard: {len(standard_chunks)} chunks")
        print(f"   Adaptive: {len(chunks)} chunks")
        
        # Show chunk size recommendations
        if len(audio_data) > 100 * 1024 * 1024:
            print(f"   💡 Large file detected - using {get_file_size_str(FALLBACK_CHUNK_SIZE)} chunks")
        elif len(audio_data) > 200 * 1024 * 1024:
            print(f"   💡 Very large file detected - using {get_file_size_str(EMERGENCY_CHUNK_SIZE)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            print(f"\n📦 Chunk {chunk_num}:")
            print(f"   Size: {get_file_size_str(len(chunk))}")
            
            # Validate chunk
            issues = validate_chunk_data(chunk, chunk_num)
            if issues:
                print(f"   ⚠️  Issues: {', '.join(issues)}")
            else:
                print(f"   ✅ Validation passed")
            
            # Show chunk header
            if len(chunk) >= 10:
                header = chunk[:10].hex()
                print(f"   Header: {header}")
                
                if chunk.startswith(b'ID3'):
                    print(f"   Format: MP3 with ID3 tag")
                elif chunk.startswith(b'RIFF'):
                    print(f"   Format: WAV/RIFF")
                elif chunk.startswith(b'\xff\xfb') or chunk.startswith(b'\xff\xfa'):
                    print(f"   Format: MP3 frame")
                else:
                    print(f"   Format: Unknown/Raw audio")
        
        # Test API call on first chunk if API key provided
        if api_key and chunks:
            print(f"\n🌐 Testing API call on first chunk...")
            test_chunk = chunks[0]
            
            try:
                url = "https://api.fish.audio/v1/asr"
                headers = {
                    "Authorization": f"Bearer {api_key.strip()}",
                    "Content-Type": "application/msgpack"
                }
                
                payload = {
                    "audio": test_chunk,
                    "ignore_timestamps": False,
                }
                
                print(f"   Payload size: {len(ormsgpack.packb(payload))} bytes")
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    data=ormsgpack.packb(payload), 
                    timeout=120
                )
                
                print(f"   Response: {response.status_code}")
                print(f"   Response size: {len(response.content)} bytes")
                
                if response.status_code == 200:
                    result = response.json()
                    text_length = len(result.get('text', ''))
                    segments_count = len(result.get('segments', []))
                    print(f"   ✅ Success: {text_length} chars, {segments_count} segments")
                    if text_length > 0:
                        preview = result.get('text', '')[:100]
                        print(f"   Preview: \"{preview}...\"")
                else:
                    print(f"   ❌ Error: {response.text[:200]}")
                    
            except Exception as e:
                print(f"   ❌ API call failed: {str(e)}")
        
    except Exception as e:
        print(f"❌ Chunking failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_chunks.py <audio_file> [api_key]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    if api_key:
        print("🔑 API key provided - will test actual API calls")
    else:
        print("📝 No API key provided - will only test chunking logic")
    
    test_chunk_processing(file_path, api_key)
