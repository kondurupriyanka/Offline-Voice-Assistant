# Offline Voice Assistant

A simple **offline voice assistant** built in Python. It can **record your voice, transcribe it using OpenAI's Whisper model, and respond using text-to-speech**. This version works completely offline on Windows and does not require PyAudio.

---

## Features

- Record audio using your microphone (`sounddevice`)  
- Transcribe speech to text using **Whisper**  
- Speak back the recognized text using **pyttsx3**  
- Works completely offline, no internet connection required  

---

![Voice Assistant Running](speech_op.jgg)

## Installation

1. Clone or download the repository.  
2. Install the required Python packages:

```bash
pip install sounddevice soundfile numpy pyttsx3 whisper librosa

 
## Usage

Run the assistant:

```bash
python speech_assistant.py
