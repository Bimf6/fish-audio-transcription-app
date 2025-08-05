#!/usr/bin/env python3.10
"""
Test script to check Fish Audio API response format
"""

import requests
import ormsgpack
import json

def test_fish_audio_api():
    api_key = "97ce09205a014871bb8ee119a921137e"
    
    # Create a minimal test audio (just a few bytes for testing)
    test_audio = b"test audio data"  # This won't work but will show us the API response format
    
    url = "https://api.fish.audio/v1/asr"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/msgpack"
    }
    
    payload = {
        "audio": test_audio,
        "ignore_timestamps": False,
    }
    
    try:
        response = requests.post(
            url, 
            headers=headers, 
            data=ormsgpack.packb(payload), 
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response Keys: {list(result.keys())}")
            print(f"Full Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error Response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fish_audio_api()