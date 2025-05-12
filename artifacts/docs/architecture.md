# Project Architecture – Audio Week 5

This document explains how the audio-centric repository is organised and how data, models and services flow together.

## Repository Structure

audio-week5/
├── data/                  # Datasets & cached features
│   ├── urbansound8k/      # Task 1
│   ├── musicnet/          # Task 2.2
│   └── personalised/      # Task 2.1 (user speech)
├── src/
│   ├── datamodules/       # TorchData / Lightning data modules
│   ├── models/            # CNN, Whisper wrapper, Music Transformer
│   ├── tasks/             # train_task1.py, finetune_whisper.py, train_music.py
│   ├── utils_audio.py     # STFT ▸ log-mel helpers, augmentation utils
│   └── infer/             # FastAPI + CLI entry points
├── notebooks/             # Exploratory analysis
├── docker/                # Containerisation (API + Streamlit)
├── playground/            # Snippets of code I'm playing around with
└── checkpoints/           # Saved weights

## Component Overview

| Layer            | Role                                                     |
|------------------|----------------------------------------------------------|
| **Datamodules**  | Unified loaders for wave → log-mel/spec; wraps augments. |
| **Models**       | *cnn_classifier.py*, *whisper_ft.py*, *music_transcriber.py*, *tiny_tts.py*. |
| **Tasks**        | Reproducible scripts called by CLI or Makefile targets.  |
| **Infer Service**| FastAPI routes: `/classify`, `/transcribe`, `/music`, `/tts`. |
| **Streamlit UI** | Interactive demo: record/upload audio, see predictions.  |

### Data Flow (Task 1 example)

1. `.wav` clip → **STFT** → **log-mel spectrogram** (`utils_audio.py`)  
2. Spectrogram → **CNN** (`models/cnn_classifier.py`)  
3. Class logits → FastAPI JSON or Streamlit chart.  

The same log-mel front-end feeds Whisper and the Music Transformer, ensuring consistent preprocessing across tasks. 