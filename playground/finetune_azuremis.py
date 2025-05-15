import torch
import whisper
import os
import sys
from pathlib import Path

# Define sample audio file path
SAMPLE_AUDIO_PATH = 'playground/data/sample.wav'

# Check if sample file exists
if not os.path.exists(SAMPLE_AUDIO_PATH):
    print(f"Error: Sample audio file not found at {SAMPLE_AUDIO_PATH}")
    print("Please provide a valid audio file for finetuning or run the following command to download a sample file:")
    print("curl -L -o playground/data/sample.wav https://github.com/openai/whisper/raw/main/tests/jfk.flac")
    sys.exit(1)

try:
    # Load model and audio
    print("Loading whisper model (tiny)...")
    model = whisper.load_model('tiny')
    
    print(f"Loading audio from {SAMPLE_AUDIO_PATH}...")
    audio = whisper.load_audio(SAMPLE_AUDIO_PATH)
    audio = whisper.pad_or_trim(audio)
    lg_ml = whisper.log_mel_spectrogram(audio)
    tknsr = whisper.tokenizer.get_tokenizer(multilingual=True)
    
    # Baseline decode
    print("Running baseline decode...")
    opt = whisper.DecodingOptions()
    res = whisper.decode(model, lg_ml.to(model.device), opt)
    print('Baseline:', res.text)
    print('-------')
    
    # Build token sequence for "Hello, my name is Bes."
    ids = []
    ids += [tknsr.sot]
    ids += [tknsr.language_token]
    ids += [tknsr.transcribe]
    ids += [tknsr.no_timestamps]
    ids += tknsr.encode(' Hello, my name is Bes.')
    ids += [tknsr.eot]
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # —– Training step —–
    print("Training step...")
    model.train()
    tks = torch.tensor(ids).unsqueeze(0).to(model.device)
    mel = whisper.log_mel_spectrogram(audio).unsqueeze(0).to(model.device)
    
    pred = model(tokens=tks, mel=mel)
    trgt = tks[:, 1:].contiguous()
    pred = pred[:, :-1, :].contiguous()
    
    print('Ids Target:', trgt.squeeze().tolist())
    print('Ids Output:', torch.argmax(pred, dim=-1).squeeze().tolist())
    print('Txt Target:', tknsr.decode(trgt.squeeze().tolist()))
    print('Txt Output:', tknsr.decode(torch.argmax(pred, dim=-1).squeeze().tolist()))
    
    loss = criterion(pred.transpose(1, 2), trgt)
    print('Loss:', loss.item())
    print('-------')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # —– Evaluation step —–
    print("Evaluation step...")
    model.eval()
    prd = model(tokens=tks, mel=mel)
    prd = prd[:, :-1, :].contiguous()
    
    print('Ids Target:', trgt.squeeze().tolist())
    print('Ids Output:', torch.argmax(prd, dim=-1).squeeze().tolist())
    print('Txt Target:', tknsr.decode(trgt.squeeze().tolist()))
    print('Txt Output:', tknsr.decode(torch.argmax(prd, dim=-1).squeeze().tolist()))
    
    loss = criterion(prd.transpose(1, 2), trgt)
    print('Loss:', loss.item())
    
    print("\nFinetuning completed successfully!")

except Exception as e:
    print(f"Error during execution: {e}")
    sys.exit(1)