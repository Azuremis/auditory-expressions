# Docker Setup Guide – Audio Week 5

The container bundles **FastAPI** (inference API) *and* a **Streamlit** audio demo.

## Quick Start

```bash
docker compose up -d          # builds + starts services
open http://localhost:8501    # Streamlit
curl  localhost:8080/health   # FastAPI ping
```

## Key Ports & Volumes

| Port | Service |
|------|---------|
| 8080 | FastAPI |
| 8501 | Streamlit UI |

| Volume | Purpose |
|--------|---------|
| ./checkpoints:/app/checkpoints | Mounts model weights |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_TASK1 | checkpoints/cnn.pt | UrbanSound CNN |
| MODEL_WHISPER | checkpoints/whisper_ft.pt | Fine-tuned Whisper |
| MODEL_MUSIC | checkpoints/music.pt | Music Transformer |
| MODEL_TTS | checkpoints/diffusion_tiny.pt | Tiny diffusion vocoder |

## Production Tips
- Split API and Streamlit into separate containers for scaling.
- Add health-checks and resource limits in docker-compose.yml.

## Docker Compose Configuration

The `docker-compose.yml` file defines the service configuration:

```yaml
version: "3.9"

services:
  app:
    build: 
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8080:8080"  # FastAPI port
      - "8501:8501"  # Streamlit port
    volumes:
      - ./checkpoints:/app/checkpoints
    environment:
      - MODEL_PATH=checkpoints/best_model.pt
    restart: unless-stopped
```

### Configuration Options

- **Ports**:
  - `8080`: The FastAPI service
  - `8501`: The Streamlit demo

- **Volumes**:
  - `./checkpoints:/app/checkpoints`: Maps the local checkpoints directory to the container

- **Environment Variables**:
  - `MODEL_PATH`: Path to the model checkpoint (relative to container)

## Building the Docker Image Manually

If you prefer to build and run the container manually:

```bash
# Build the image
docker build -t vit-mnist -f docker/Dockerfile .

# Run the container
docker run -p 8080:8080 -p 8501:8501 -v $(pwd)/checkpoints:/app/checkpoints vit-mnist
```

## Dockerfile Explained

The `docker/Dockerfile` includes:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir streamlit-drawable-canvas

# Copy the entire project
COPY . .

# Make sure the checkpoints directory exists
RUN mkdir -p checkpoints

# Expose ports for FastAPI and Streamlit
EXPOSE 8080 8501

# Start both services using Supervisord
COPY docker/supervisord.conf /etc/supervisord.conf
RUN pip install --no-cache-dir supervisor

# Start supervisord
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
```

Key aspects:
- Based on Python 3.9 slim image for smaller size
- Installs all dependencies including the Streamlit drawing canvas
- Uses Supervisord to manage multiple services in one container
- Exposes ports for both services

## Supervisord Configuration

Supervisord manages both services in the same container:

```ini
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[program:api]
command=uvicorn docker.src.inference.app:app --host 0.0.0.0 --port 8080
directory=/app
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/api.log
stderr_logfile=/var/log/supervisor/api_error.log
environment=PYTHONPATH=/app

[program:streamlit]
command=streamlit run docker/src/streamlit/app.py
directory=/app
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/streamlit.log
stderr_logfile=/var/log/supervisor/streamlit_error.log
environment=PYTHONPATH=/app
```

## Using the Deployed Services

### API Usage

```bash
# Health check
curl http://localhost:8080/health

# Predict from image file
curl -X POST -F "file=@path/to/image.png" http://localhost:8080/predict/image
```

### Demo Usage

Open http://localhost:8501 in a web browser to access the interactive demo.

## Production Deployment Considerations

For production deployments, consider these adjustments:

1. **Separate Containers**: Split API and Streamlit into separate containers for better scalability
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/api.Dockerfile
       ports:
         - "8080:8080"
     
     streamlit:
       build:
         context: .
         dockerfile: docker/streamlit.Dockerfile
       ports:
         - "8501:8501"
   ```

2. **Health Checks**: Add health checks for better container orchestration
   ```yaml
   services:
     app:
       # ...
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

3. **Environment Variables**: Externalize configuration using environment variables
   ```yaml
   services:
     app:
       # ...
       env_file:
         - .env.production
   ```

4. **Resource Limits**: Set resource constraints
   ```yaml
   services:
     app:
       # ...
       deploy:
         resources:
           limits:
             cpus: '0.50'
             memory: 512M
   ```

5. **Logging**: Configure proper logging
   ```yaml
   services:
     app:
       # ...
       logging:
         driver: "json-file"
         options:
           max-size: "10m"
           max-file: "3"
   ```

## Troubleshooting

### Container Won't Start

Check the logs:
```bash
docker-compose logs app
```

Common issues:
- Missing model checkpoint - ensure the checkpoint exists in `./checkpoints/`
- Port conflicts - change port mappings if 8080 or 8501 are in use

### API Returns 503 Service Unavailable

This typically means the model failed to load. Check:
- Model checkpoint path is correct in environment variables
- The checkpoint file exists and is valid

### Streamlit Shows Connection Error

If Streamlit can't connect to the API:
- Ensure both services are running
- Check that the API health endpoint returns 200 OK
- Verify network settings in the container 