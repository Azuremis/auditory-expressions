# Configuration Guide – Audio Week 5

This document explains the YAML configuration files for all audio processing tasks.

## Default Configuration

```yaml
# Default YAML configuration – Audio Week 5

device: "cuda"                 # cuda / cpu / mps
data_root: "./data"

# --- Feature parameters ----------------------------------------------------
sample_rate: 22050             # Hz
n_mels: 128
window_ms: 50
hop_ms: 25

# --- Task-specific blocks ---------------------------------------------------
cnn:
  model_name: "resnet18"
num_classes: 10
  dropout: 0.3

whisper:
  checkpoint: "tiny.en"
  lr: 1.0e-4
  lora_rank: 8
  freeze_encoder: true

music:
  encoder_dim: 512
  encoder_layers: 4
  seq_reduction: 8            # pooled time reduction factor
  lr: 1.0e-4

tts:
  tacotron_dim: 256
  vocoder_ckpt: "./checkpoints/diffusion_tiny.pt"

# --- Training hyper-params --------------------------------------------------
batch_size: 32
epochs: 25
weight_decay: 1.0e-4
use_scheduler: true

# --- Logging ----------------------------------------------------------------
wandb:
  enable: true
  project: "audio-week5"
  group: "default"

checkpoints_dir: "./checkpoints"
```

## Parameter Descriptions

### Audio Feature Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `sample_rate` | Audio sampling frequency in Hz | `22050` |
| `n_mels` | Number of mel filterbanks | `128` |
| `window_ms` | Analysis window size in milliseconds | `50` |
| `hop_ms` | Window hop size in milliseconds | `25` |

### Task-Specific Parameters

#### UrbanSound CNN
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `cnn.model_name` | Model architecture | `resnet18` |
| `cnn.num_classes` | Number of sound classes | `10` |
| `cnn.dropout` | Dropout rate | `0.3` |

#### Whisper Fine-Tuning
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `whisper.checkpoint` | Base Whisper model | `tiny.en` |
| `whisper.lr` | Learning rate | `1.0e-4` |
| `whisper.lora_rank` | LoRA adapter rank | `8` |
| `whisper.freeze_encoder` | Whether to freeze encoder | `true` |

#### Music Transcription
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `music.encoder_dim` | Encoder dimension | `512` |
| `music.encoder_layers` | Number of encoder layers | `4` |
| `music.seq_reduction` | Time reduction factor | `8` |
| `music.lr` | Learning rate | `1.0e-4` |

#### Tiny TTS
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `tts.tacotron_dim` | Tacotron model dimension | `256` |
| `tts.vocoder_ckpt` | Path to vocoder checkpoint | `./checkpoints/diffusion_tiny.pt` |

### Training Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `batch_size` | Batch size for training | `32` |
| `epochs` | Number of training epochs | `25` |
| `weight_decay` | Weight decay coefficient | `1.0e-4` |
| `use_scheduler` | Whether to use LR scheduler | `true` |

## Command Line Usage

To train with specific configuration overrides:

```bash
# Task 1 - UrbanSound Classifier
python train_task1.py cnn.dropout=0.5 batch_size=64

# Task 2.1 - Whisper Fine-tuning
python finetune_whisper.py whisper.freeze_encoder=false whisper.lr=5e-5

# Task 2.2 - Music Transcription
python train_music.py music.encoder_layers=6 epochs=30 