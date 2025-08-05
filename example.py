from fish_audio_sdk import Session, ASRRequest

session = Session("your_api_key")

# Read the audio file
with open("input_audio.mp3", "rb") as audio_file:
    audio_data = audio_file.read()

# Option 1: Without specifying language (auto-detect)
response = session.asr(ASRRequest(audio=audio_data))

# Option 2: Specifying the language
response = session.asr(ASRRequest(audio=audio_data, language="en"))

# Option 3: With precise timestamps (may increase latency for short audio)
response = session.asr(ASRRequest(audio=audio_data, ignore_timestamps=False))

print(f"Transcribed text: {response.text}")
print(f"Audio duration: {response.duration} seconds")

for segment in response.segments:
    print(f"Segment: {segment.text}")
    print(f"Start time: {segment.start}, End time: {segment.end}")