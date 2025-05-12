# GPU Acceleration – Audio Week 5

Proper GPU use slashes spectrogram and model runtimes.

## Fast Path

```python
device = torch.device(
    'cuda' if torch.cuda.is_available()
    else ('mps' if torch.backends.mps.is_available() else 'cpu')
)
model.to(device)
```

## Mixed Precision

Whisper & Music Transformer happily run with AMP:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    logits = model(mel_batch)
    loss = criterion(logits, labels)
```

## Chunked Inference (Long Audio)

```python
def chunk_predict(wave, chunk_s=30, overlap_s=5):
    # split, predict with model, stitch logits / tokens back together
```

## Multi-GPU

Use PyTorch Lightning's ddp or plain DistributedDataParallel for the music model; Whisper fine-tune is light enough for a single GPU. 