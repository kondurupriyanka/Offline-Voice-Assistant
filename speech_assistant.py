import whisper
import pyttsx3
import numpy as np
import soundfile as sf
import sounddevice as sd
import wave
import io
import librosa

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize TTS engine
tts_engine = pyttsx3.init()

def speak(text):
    """Speak text using pyttsx3"""
    print(f"Assistant: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def record_command(duration=5, sample_rate=16000):
    """Record audio from microphone using sounddevice"""
    print(f"🎙️ Recording {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    
    # Convert to WAV buffer
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    wav_buffer.seek(0)
    return wav_buffer

def transcribe_audio(wav_buffer):
    """Transcribe audio using Whisper"""
    audio_array, sample_rate = sf.read(wav_buffer, dtype="float32")
    
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    if sample_rate != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    
    audio_array = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(audio_array).to(whisper_model.device)
    options = whisper.DecodingOptions(language="en", fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    return result.text.strip()

if __name__ == "__main__":
    print("🟢 Offline Voice Assistant started")
    while True:
        try:
            wav = record_command(duration=5)  # record 5 seconds
            text = transcribe_audio(wav)
            print("You said:", text)
            speak("You said: " + text)
        except KeyboardInterrupt:
            print("👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
