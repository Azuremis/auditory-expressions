# Training Guide – Audio Week 5

## Task 1: UrbanSound CNN

```bash
python train_task1.py \
       data.root=./data/urbansound8k \
       trainer.max_epochs=30
```

Key flags: model.lr, augment.spec_augment=true.

## Task 2.1: Whisper Fine-Tune

```bash
python finetune_whisper.py \
       experiment=C \
       data.dir=./data/personalised
```

Supports freeze_encoder, lora_rank, prompt_tokens.

## Task 2.2: Music Transcription

```bash
python train_music.py \
       data.dir=./data/musicnet \
       model.encoder_layers=4
```

## Extra: Tiny TTS

   ```bash
python train_tacotron.py
python train_diffusion_vocoder.py
```

Log all runs with wandb=true. 