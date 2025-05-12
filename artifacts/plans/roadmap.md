# Audio Week 5: Development Roadmap

This roadmap provides a single-page navigation guide for the five-day sprint (12 May → 16 May 2025). Pair it with the detailed Handbook for code snippets and FAQs.

## Sprint Timeline Overview

### Day 1 – Monday (May 12)
| Task | Duration | Status | Dependencies |
|------|----------|--------|--------------|
| Repo & Environment Setup | Morning (0.5d) | ✅ Complete | - |
| Download UrbanSound 8K | Morning (0.3d) | 🔄 In Progress | - |
| Spectrogram Extraction | Afternoon (0.2d) | ⏳ Not Started | Download data |
| Start CNN Baseline Training | Afternoon - Evening (0.5d) | ⏳ Not Started | Spectrogram extraction |

### Day 2 – Tuesday (May 13)
| Task | Target Time | Status | Dependencies |
|------|-------------|--------|--------------|
| Continue CNN Training | Morning (1.0d) | ⏳ Not Started | Started on Day 1 |
| Baseline ResNet-18 val acc ≥ 70% | 13:00 | ⏳ Not Started | Spectrogram cache |
| Confusion Matrix Generation | 14:00 | ⏳ Not Started | Baseline accuracy |
| Draft Model Card | 17:00 | ⏳ Not Started | Baseline accuracy |

### Day 3 – Wednesday (May 14)
| Task | Duration | Status | Dependencies |
|------|----------|--------|--------------|
| Whisper Experiments A-C | Morning (0.6d) | ⏳ Not Started | Personal audio set |
| Whisper Experiments D-F | Afternoon (0.4d) | ⏳ Not Started | Experiments A-C |
| WER Table + Audio Demos | Evening (0.3d) | ⏳ Not Started | Experiments D-F |
| All Whisper Experiments Done | 18:00 | ⏳ Not Started | Personal audio set |

### Day 4 – Thursday (May 15)
| Task | Duration | Status | Dependencies |
|------|----------|--------|--------------|
| MusicNet Dataset Curation | Morning (0.5d) | ⏳ Not Started | - |
| Music Transformer Training | Afternoon (0.8d) | ⏳ Not Started | MusicNet tokens |
| F1 Evaluation + Demo Render | Evening (0.3d) | ⏳ Not Started | Transformer training |
| Music Model F1 Complete | 18:00 | ⏳ Not Started | MusicNet tokens |

### Day 5 – Friday (May 16)
| Task | Duration | Status | Dependencies |
|------|----------|--------|--------------|
| Diffusion Vocoder Build | Morning-Afternoon (0.7d) | ⏳ Not Started | - |
| TTS Demo Complete | 15:00 | ⏳ Not Started | Diffusion vocoder |
| Slide Deck & Wrap-Up | Afternoon (0.3d) | ⏳ Not Started | Vocoder build |
| Repository Audit & Tag v1.0 | 18:00 | ⏳ Not Started | All tasks complete |

## Phase 0 – Setup (Mon AM)

- ✅ Repo & Environment
  - poetry install, pre-commit hooks, FP16 enabled.
- 🔄 Data Download
  - UrbanSound 8K → data/urbansound8k/raw (6.36 GB, mirror #2).
- ⏳ Spectrogram Cache
  - extract_spectro.py – 128-mel, 25 ms hop, 1024 FFT.

## Phase 1 – Task 1: Environmental-Sound Classifier (Mon PM → Tue)

### Milestones

| Checkpoint | Target | Owner | Status |
|------------|--------|-------|--------|
| Baseline ResNet-18 val acc ≥ 70 % | Tue 13:00 | @audio-lead | ⏳ Not Started |
| Confusion matrix saved | Tue 14:00 | @ml-ops | ⏳ Not Started |
| Model card drafted | Tue 17:00 | @docs | ⏳ Not Started |

### Deliverables
- train.py + Hydra config urbansound.yaml.
- metrics_task1.json & confusion_matrix.png committed.

## Phase 2 – Task 2.1: Personalised Whisper (Wed)

### Experiment Matrix

| ID | Variant | Goal | Status |
|----|---------|------|--------|
| A | Encoder frozen | Avoid forgetting | ⏳ Not Started |
| B | Full-model train | Max WER drop | ⏳ Not Started |
| C | LoRA-rank 8 | Memory-efficient | ⏳ Not Started |
| D | `<speaker>` | transcribe | ⏳ Not Started |
| E | Prompt with rare name | Recall boost | ⏳ Not Started |
| F | Weight-drift stats | Diagnose forgetting | ⏳ Not Started |

Run python finetune.py experiment=<ID>. Log to W&B run group whisper_finetune.

### Completion Criteria
- WER ↓ ≥ 30 % on personalised set.
- General-speech WER degradation < 5 %.
- Notebook whisper_report.ipynb shows before/after transcripts and audio.

## Phase 3 – Task 2.2: Music Note Transcription (Thu)

### Steps
1. **Data** – Script download_musicnet.sh (33 GB). Downsample → 22 kHz. ⏳
2. **Tokens** – Piano-roll encoding (pitch, velocity, duration). ⏳
3. **Model** – Conv-stride (2×3) + 4-layer Transformer (d = 512). ⏳
4. **Training** – 25 epochs, Adam 1e-4, label-smoothing 0.1. ⏳
5. **Eval** – Note-on F1 ≥ 0.55; render 30 s MIDI via FluidSynth. ⏳

## Phase 4 – Extra: Tiny TTS Reproduction (Fri)

### Goals
- LJ-Speech one-sentence demo pronouncing "live" vs "leave" correctly. ⏳
- 10-slide deck summarising: pipeline, results, challenges, next steps. ⏳

### Key Tasks
1. Implement g2p via g2p_en. ⏳
2. Train tacotron-lite (3 M params) to mels. ⏳
3. Port tiny diffusion vocoder weights from Xie et al. appendix. ⏳

## Timeline & Dependencies

| Milestone | Date/Time | Depends on | Status |
|-----------|-----------|------------|--------|
| Task 1 baseline acc | Tue 13:00 | Spectrogram cache | ⏳ Not Started |
| Task 1 card | Tue 17:00 | Baseline acc | ⏳ Not Started |
| Whisper experiments done | Wed 18:00 | Personal audio set | ⏳ Not Started |
| Music model F1 done | Thu 18:00 | MusicNet tokens | ⏳ Not Started |
| TTS demo | Fri 15:00 | Diffusion vocoder build | ⏳ Not Started |
| Repo audit & tag v1.0 | Fri 18:00 | All above | ⏳ Not Started |

## Resource Allocation (5-person squad)

| Name | Mon-Tue | Wed | Thu | Fri |
|------|---------|-----|-----|-----|
| @audio-lead | CNN training | Whisper exp | Music transcript | TTS build |
| @ml-ops | Data pipelines | WandB boards | Token scripts | Deck polish |
| @docs | Handbook edits | Whisper report | MIDI render doc | Final audit |
| @qa | Test harness | WER validator | F1 scripts | Smoke tests |
| @pm | Stand-ups | Check-ins | Risk log | Release tag |

## Success Metrics
1. **Accuracy** – UrbanSound val acc ≥ 70 %.
2. **WER** – Personalised ↓ ≥ 30 % while Base ↓ ≤ 5 %.
3. **Music F1** – Note-on F1 ≥ 0.55.
4. **TTS Demo** – "live" vs "leave" pronounced context-correctly.
5. **Reproducibility** – Any teammate reproduces results in ≤ 60 min on single RTX 4090.

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Catastrophic forgetting | High | Medium | Freeze encoder, small LR, weight-drift logs |
| GPU OOM on music model | Medium | Medium | Conv-stride, FP16, gradient checkpointing |
| Data imbalance (car horn) | Med | High | Focal loss γ = 2, oversample |
| Friday overrun | High | Medium | Early slide draft, scope lock Wed PM |

## Quick Commands

```bash
# Task 1
python train.py task=urbansound hydra.run.dir=outputs/task1

# Whisper fine-tune (Experiment C)
python finetune.py experiment=C wandb.group=whisper_finetune

# Music transcription
python train_music.py config=musicnet_base epochs=25

# Diffusion vocoder inference
python tts_infer.py --text "I live in Leeds" --ckpt checkpoints/diffusion_tiny.pt
```

Print this roadmap, tick boxes as you go, and enjoy the sprint! 🎧🚀