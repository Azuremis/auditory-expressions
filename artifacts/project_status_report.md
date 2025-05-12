# Audio Week 5: Project Status Report

![Project Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen)
![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-blue)
![Documentation: Ready](https://img.shields.io/badge/Documentation-Ready-brightgreen)

**Last Updated:** May 12, 2025

## Executive Dashboard

| Task | Status | Timeline |
|------|--------|----------|
| **Phase 0: Setup** | ✅ Complete | May 12 (AM) |
| **Task 1: UrbanSound Classifier** | 🔄 Not Started | May 12-13 |
| **Task 2.1: Whisper Fine-Tune** | 🔄 Not Started | May 14 |
| **Task 2.2: Music Transcription** | 🔄 Not Started | May 15 |
| **Extra: Tiny TTS** | 🔄 Not Started | May 16 |

| Component | Status | Trend |
|-----------|--------|-------|
| **Environment Setup** | ⭐⭐⭐⭐⭐ 5/5 | ✓ Complete |
| **Documentation** | ⭐⭐⭐⭐⭐ 5/5 | ✓ Complete |
| **Data Preparation** | ⭐⭐☆☆☆ 2/5 | 🔄 In Progress |
| **Model Development** | ⭐☆☆☆☆ 1/5 | 🔄 Not Started |
| **Code Quality** | ⭐⭐⭐☆☆ 3/5 | → Stable |
| **Test Coverage** | ⭐☆☆☆☆ 1/5 | 🔄 Not Started |

### Recent Milestones

✅ Repository & environment setup complete  
✅ Documentation framework established  
✅ Project roadmap finalized  
⌛ UrbanSound 8K dataset download  
⌛ Spectrogram extraction pipeline  

### Current Focus

🔍 Download UrbanSound 8K dataset  
🔍 Extract log-mel spectrograms  
🔍 Implement baseline ResNet-18 CNN model  
🔍 Setup WandB logging  

## Executive Summary

The "Audio Week 5" project is a 5-day sprint (May 12-16, 2025) focused on implementing a suite of audio processing models. This project demonstrates various audio ML tasks, from environmental sound classification to speech recognition, music transcription, and text-to-speech synthesis.

The project has been initialized with:

- A clean repository structure with proper organization
- Comprehensive documentation covering all planned tasks
- Environment setup with all required dependencies
- Sprint planning with clear milestones and deliverables

Current progress includes completing the repository setup and documentation. The next immediate steps involve data acquisition, feature extraction, and beginning work on the first task (UrbanSound CNN classifier).

## Project Overview

### Core Purpose

This project demonstrates the versatility of deep learning approaches to audio processing by implementing four distinct audio tasks:

1. **Environmental Sound Classification**: CNN model for classifying urban sounds (car horn, dog bark, etc.)
2. **Personalized Speech Recognition**: Fine-tuning Whisper for improved recognition of specific voices/names
3. **Music Note Transcription**: Converting music audio into MIDI-like note sequences
4. **Text-to-Speech Synthesis**: Generating natural speech from text input

The variety of tasks showcases different techniques in deep learning for audio, from spectrograms and CNNs to transformer models and diffusion vocoders.

### Key Features

- **Modular Design**: Separate components for different audio tasks
- **Config-Driven**: YAML configuration files for model and training parameters
- **Interactive Demo**: Streamlit web interface for audio processing
- **API Service**: FastAPI-based services for all audio tasks
- **Docker Deployment**: Ready-to-run containers for demo and API
- **MLX Support**: Optional Apple Silicon optimization

## Current State

### Functionality Status

| Component | Status | Notes |
|-----------|--------|-------|
| Repository Structure | ✅ Complete | Basic folder organization set up |
| Environment Setup | ✅ Complete | Dependencies installed, virtual env ready |
| Documentation | ✅ Complete | All documentation files created and updated |
| Data Acquisition | 🚧 In Progress | UrbanSound 8K download planned |
| Feature Extraction | 🔄 Not Started | Log-mel spectrogram extraction |
| UrbanSound Classifier | 🔄 Not Started | ResNet-18 based CNN |
| Whisper Fine-tuning | 🔄 Not Started | Scheduled for May 14 |
| Music Transcription | 🔄 Not Started | Scheduled for May 15 |
| Tiny TTS | 🔄 Not Started | Scheduled for May 16 |
| Docker Deployment | 🔄 Not Started | After models are complete |
| Streamlit Demo | 🔄 Not Started | After models are complete |

### Architecture

The codebase follows a well-structured organization:

```
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
└── checkpoints/           # Saved weights
```

### Task-Specific Status

#### Task 1: Environmental-Sound Classifier (May 12-13)

- ✅ Documentation prepared
- 🚧 UrbanSound 8K dataset download in progress
- 🔄 Spectrogram extraction not started
- 🔄 CNN model training not started
- 🔄 Evaluation and model card not started

**Next Immediate Steps:**
- Complete dataset download
- Implement spectrogram extraction
- Train baseline ResNet-18 model

#### Task 2.1: Personalised Whisper (May 14)

- ✅ Documentation prepared
- 🔄 Personalised speech dataset not acquired
- 🔄 Whisper fine-tuning experiments not started
- 🔄 WER evaluation not started

**Next Immediate Steps:**
- Prepare for Experiment A (encoder frozen)
- Set up WandB logging for experiments

#### Task 2.2: Music Transcription (May 15)

- ✅ Documentation prepared
- 🔄 MusicNet dataset not acquired
- 🔄 Transformer model not implemented
- 🔄 Note-wise F1 evaluation not set up

**Next Immediate Steps:**
- Write download script for MusicNet
- Begin designing token representation

#### Extra: Tiny TTS (May 16)

- ✅ Documentation prepared
- 🔄 Diffusion vocoder not implemented
- 🔄 Tacotron-lite not implemented
- 🔄 Demo not created

**Next Immediate Steps:**
- Setup LJ-Speech dataset preparation script

## Next Steps

Based on the current status, the following next steps are planned:

### Immediate (Next 24 Hours)

1. Download UrbanSound 8K dataset
2. Extract log-mel spectrograms
3. Setup WandB logging
4. Implement basic CNN model

### Short Term (This Week)

1. Complete Task 1 (UrbanSound CNN) by May 13
2. Run Whisper fine-tuning experiments by May 14
3. Implement music transcription by May 15
4. Build Tiny TTS prototype by May 16

### Success Criteria

The project will be considered successful if we meet these targets:

1. Task 1: UrbanSound validation accuracy ≥ 70%
2. Task 2.1: WER reduction ≥ 30% on personalized set, ≤ 5% degradation on general speech
3. Task 2.2: Note-on F1 score ≥ 0.55
4. Extra: TTS demo correctly pronouncing "live" vs "leave" based on context

## Challenges and Risks

The project faces several challenges and risks:

1. **Tight Timeline**: 5-day sprint requires careful time management
2. **Catastrophic Forgetting**: Whisper fine-tuning may degrade general performance
3. **GPU Memory Constraints**: Music transformer may exceed available memory
4. **Data Imbalance**: UrbanSound dataset has class imbalance (car horn, etc.)

## Conclusion

The Audio Week 5 project has successfully completed its initial setup phase with repository initialization and comprehensive documentation. As we move forward with the actual task implementations, the focus will be on maintaining the rapid development pace required by the 5-day sprint timeline while ensuring we meet the defined success criteria.

The clear structure and detailed documentation provide a solid foundation for the development work ahead. Daily progress tracking will be essential to ensure we complete all planned tasks within the sprint timeframe. 