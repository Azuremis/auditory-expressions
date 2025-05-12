# Decoder Components in This Repo

* **Music Transformer** – sequence-to-sequence decoder that outputs MIDI-style note tokens conditioned on log-mel encodings.
* **Tacotron-Lite** – text-to-mel decoder for the extra TTS task.
* Cross-attention integrates encoder outputs (spectrogram features) with autoregressive token generation.

See `models/music_transcriber.py` and `models/tacotron.py` for implementation details. 