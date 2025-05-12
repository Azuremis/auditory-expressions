# MLX Optimisation – Apple Silicon (Optional Stretch)

Running Whisper tiny + CNN on MacBooks? MLX can cut times ~2×.

* **Core idea**: keep PyTorch models, but move log-mel & CNN layers to MLX primitives (`mlx.nn.Conv2d`, etc.).
* **Scripts**: `src/convert/torch_to_mlx.py`, `train_mlx.py`.
* **Benchmarks (M3 Pro)**  
  * PyTorch CNN: 95 it/s → MLX CNN: 210 it/s  
  * Whisper tiny decode: 130 ms → 62 ms 