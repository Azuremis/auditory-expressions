# Inference Guide – Audio Week 5

Four endpoints, one CLI.

## 1. Command-Line

```bash
# Task 1 – sound classification
python -m src.infer.cli classify \
       --model checkpoints/cnn.pt \
       --audio sample.wav

# Task 2.1 – personalised transcription
python -m src.infer.cli transcribe \
       --model checkpoints/whisper_ft.pt \
       --audio speech.wav

# Task 2.2 – music note transcription
python -m src.infer.cli music \
       --model checkpoints/music.pt \
       --audio piano.wav
       
# Extra – text-to-speech
python -m src.infer.cli tts \
       --model checkpoints/tacotron.pt \
       --vocoder checkpoints/diffusion_tiny.pt \
       --text "I live in Leeds"
```

## 2. Python API

```python
from src.infer.api import Classifier, Transcriber, MusicTranscriber, TinyTTS

# Task 1 - Sound classification
pred = Classifier("checkpoints/cnn.pt").predict("alarm.wav")
# Returns: {"class_id": 3, "class_name": "car_horn", "confidence": 0.89}

# Task 2.1 - Speech transcription
text = Transcriber("checkpoints/whisper_ft.pt").transcribe("speech.wav")
# Returns: "the quick brown fox jumps over the lazy dog"

# Task 2.2 - Music transcription
notes = MusicTranscriber().predict("piano.wav")
# Returns: [{"pitch": 60, "start": 0.5, "duration": 0.25}, ...]

# Extra - Text-to-speech
wave = TinyTTS().synthesise("hello world")
# Returns: np.array of audio samples (waveform)
```

## 3. REST API (FastAPI)

| Route | Method | Body | Response |
|-------|--------|------|----------|
| `/health` | GET | – | `{"status": "ok"}` |
| `/classify` | POST | multipart file | `{"class_id": 3, "class_name": "car_horn", "confidence": 0.89}` |
| `/transcribe` | POST | multipart file | `{"text": "the quick brown fox..."}` |
| `/music` | POST | multipart file | `{"notes": [{"pitch": 60, "start": 0.5, "duration": 0.25}, ...]}` |
| `/tts` | POST | JSON `{"text": "hello"}` | Audio file (WAV) |

## 4. Batch Processing

For processing multiple audio files:

```python
# Batch classification
audio_files = ["car1.wav", "car2.wav", "dog1.wav"]
classifier = Classifier("checkpoints/cnn.pt")
results = [classifier.predict(file) for file in audio_files]

# Batch transcription (with progress bar)
from tqdm import tqdm
transcriber = Transcriber("checkpoints/whisper_ft.pt")
results = [transcriber.transcribe(file) for file in tqdm(audio_files)]
```

## 5. Performance Optimization

### Chunked Processing for Long Audio

For long audio files (>30s), use chunking for better memory efficiency:

```python
from src.infer.utils import chunk_audio

# Process long file in 10-second chunks with 1-second overlap
chunks = chunk_audio("long_speech.wav", chunk_s=10, overlap_s=1)
transcriber = Transcriber("checkpoints/whisper_ft.pt")
results = [transcriber.transcribe(chunk) for chunk in chunks]
transcript = " ".join(results)
```

### GPU Acceleration

All inference endpoints support GPU acceleration:

```python
# Explicitly select device
classifier = Classifier(
    model_path="checkpoints/cnn.pt",
    device="cuda"  # or "mps" for Apple Silicon, "cpu" for CPU
)
```

## 6. Interactive Demo

The project includes a Streamlit demo with interactive audio processing.
See [Streamlit Demo](streamlit-demo.md) for details.

To launch:

```bash
docker compose up -d
# Then open http://localhost:8501 in your browser
``` 