# Azure Deployment Guide

Deploy Starship Horizons Learning AI to Azure with GPU support for Whisper transcription and Ollama LLM reports.

## What Gets Deployed

- **GPU VM** (Standard_NC4as_T4_v3) with NVIDIA T4
- **Docker** with NVIDIA Container Toolkit
- **Whisper** for audio transcription (runs on GPU)
- **Ollama** for LLM-powered mission reports (runs on GPU)
- **FastAPI web server** for the analysis interface

## Prerequisites

1. **Azure CLI** - [Install Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
2. **Azure Subscription** with GPU quota (request quota increase for NC-series if needed)
3. **SSH Key Pair** - Generate if you don't have one:
   ```bash
   ssh-keygen -t ed25519 -C "azure-vm"
   ```

## Quick Start

### Deploy with the script

```bash
cd deploy/azure
chmod +x deploy.sh

# Basic deployment (will prompt for confirmation)
./deploy.sh -k ~/.ssh/id_rsa.pub -r https://github.com/YOUR_USER/SH-Learning-AI.git

# With custom models
./deploy.sh \
  -k ~/.ssh/id_rsa.pub \
  -r https://github.com/YOUR_USER/SH-Learning-AI.git \
  -w medium \
  -o qwen2.5:7b
```

### After Deployment

1. **Wait for cloud-init** (5-10 minutes):
   ```bash
   ssh azureuser@YOUR_VM_FQDN
   cloud-init status --wait
   ```

2. **Start the application**:
   ```bash
   ~/start-app.sh
   ```
   This will:
   - Build the Docker image (~5 min)
   - Download Whisper model (~1-2 min)
   - Start Ollama and download LLM model (~2-5 min depending on model)
   - Start the web server

3. **Monitor startup**:
   ```bash
   docker logs -f sh-learning-ai
   ```

   You'll see:
   ```
   [INFO] === Starship Horizons Learning AI ===
   [INFO] Starting Ollama server...
   [INFO] Ollama server is ready
   [INFO] Downloading Ollama model: llama3.2
   [INFO] Model 'llama3.2' downloaded successfully
   [INFO] Starting web server on port 8000...
   ```

4. **Access the web interface**:
   ```
   http://YOUR_VM_FQDN:8000
   ```

## Configuration Options

### Deploy Script Options

| Option | Default | Description |
|--------|---------|-------------|
| `-g, --resource-group` | sh-learning-ai-rg | Azure resource group name |
| `-l, --location` | eastus | Azure region |
| `-k, --ssh-key` | ~/.ssh/id_rsa.pub | Path to SSH public key |
| `-r, --repo` | (none) | Git repository URL to clone |
| `-s, --vm-size` | Standard_NC4as_T4_v3 | Azure VM size |
| `-w, --whisper-model` | medium | Whisper model size |
| `-o, --ollama-model` | llama3.2 | Ollama LLM model |

### Model Options

**Whisper Models** (audio transcription):
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| tiny | ~75MB | Fastest | Basic |
| base | ~150MB | Fast | Good |
| small | ~500MB | Medium | Better |
| **medium** | ~1.5GB | Slower | Great (recommended) |
| large-v3 | ~3GB | Slowest | Best |

**Ollama Models** (LLM reports):
| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| **llama3.2** | ~2GB | Fast | Good (recommended for start) |
| qwen2.5:7b | ~4GB | Medium | Better |
| qwen2.5:14b-instruct | ~9GB | Slower | Best |
| mistral | ~4GB | Medium | Good |

## VM Sizes and Pricing

| Size | GPU | vCPUs | RAM | ~Monthly Cost |
|------|-----|-------|-----|---------------|
| **Standard_NC4as_T4_v3** | 1x T4 (16GB) | 4 | 28 GB | ~$380 |
| Standard_NC8as_T4_v3 | 1x T4 (16GB) | 8 | 56 GB | ~$760 |
| Standard_NC6s_v3 | 1x V100 (16GB) | 6 | 112 GB | ~$900 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Azure VM (GPU)                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Docker Container                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │  │
│  │  │   Ollama    │  │   Whisper   │  │   FastAPI    │  │  │
│  │  │  (LLM API)  │  │ (Transcribe)│  │ (Web Server) │  │  │
│  │  │  Port 11434 │  │   (Library) │  │  Port 8000   │  │  │
│  │  └─────────────┘  └─────────────┘  └──────────────┘  │  │
│  │         │                │                │           │  │
│  │         └────────────────┴────────────────┘           │  │
│  │                     NVIDIA T4 GPU                     │  │
│  └───────────────────────────────────────────────────────┘  │
│                              │                               │
│                         Port 8000                            │
└──────────────────────────────┼──────────────────────────────┘
                               │
                          Internet
                               │
                        ┌──────┴──────┐
                        │   Browser   │
                        │  (Web UI)   │
                        └─────────────┘
```

## Useful Commands

### On the VM

```bash
# Check system status
~/check-status.sh

# View application logs
docker logs -f sh-learning-ai

# Check GPU status
nvidia-smi

# Restart the application
docker restart sh-learning-ai

# Rebuild and restart
~/start-app.sh
```

### From Your Local Machine

```bash
# Stop VM (saves money when not in use)
az vm deallocate -g sh-learning-ai-rg -n sh-learning-ai-vm

# Start VM
az vm start -g sh-learning-ai-rg -n sh-learning-ai-vm

# Delete everything
az group delete -n sh-learning-ai-rg
```

## Troubleshooting

### Cloud-init didn't complete
```bash
# Check cloud-init logs
sudo cat /var/log/cloud-init-output.log

# Re-run cloud-init (if needed)
sudo cloud-init clean && sudo cloud-init init
```

### GPU not detected
```bash
# Check NVIDIA driver
nvidia-smi

# If not working, reinstall
sudo apt-get install --reinstall nvidia-driver-535
sudo reboot
```

### Docker GPU issues
```bash
# Reconfigure NVIDIA container toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test GPU in Docker
docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Ollama model download failed
```bash
# SSH into the container
docker exec -it sh-learning-ai bash

# Manually pull the model
ollama pull llama3.2
```

### Application won't start
```bash
# Check container logs
docker logs sh-learning-ai

# Check if port is in use
sudo netstat -tlnp | grep 8000

# Rebuild from scratch
docker stop sh-learning-ai
docker rm sh-learning-ai
~/start-app.sh
```

## Files

```
deploy/azure/
├── azuredeploy.json           # ARM template
├── azuredeploy.parameters.json # Default parameters
├── deploy.sh                   # Deployment script
└── README.md                   # This file

deploy/
└── docker-entrypoint.sh        # Container startup script

Dockerfile                      # Production Docker image
.dockerignore                   # Docker build exclusions
```

## Security Recommendations

1. **Restrict IP access**: Use `-a` flag or edit `allowedSourceIPs` parameter
2. **Use SSH keys**: Never use password authentication
3. **Enable HTTPS**: Set up nginx reverse proxy with Let's Encrypt
4. **Keep updated**: Regularly update the VM and Docker images
