#!/usr/bin/env python3.10
"""
Simple test script to verify the Fish Audio Transcription app components work correctly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
        
        from fish_audio_sdk import Session, ASRRequest
        print("✓ Fish Audio SDK imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_session_creation():
    """Test that a Session can be created (without making API calls)."""
    try:
        from fish_audio_sdk import Session
        session = Session("test_key")
        print("✓ Session created successfully")
        return True
    except Exception as e:
        print(f"✗ Session creation error: {e}")
        return False

def test_asr_request_creation():
    """Test that ASRRequest can be created."""
    try:
        from fish_audio_sdk import ASRRequest
        request = ASRRequest(audio=b"test", language="en")
        print("✓ ASRRequest created successfully")
        return True
    except Exception as e:
        print(f"✗ ASRRequest creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Fish Audio Transcription App Components")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_session_creation,
        test_asr_request_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The app should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 