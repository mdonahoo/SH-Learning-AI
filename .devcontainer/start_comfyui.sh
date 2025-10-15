#!/bin/bash
# Start ComfyUI server for image generation

echo "🚀 Starting ComfyUI Server..."
echo "================================"
echo ""

# Check if ComfyUI directory exists
if [ ! -d "/workspaces/SH-Learning-AI/ComfyUI" ]; then
    echo "❌ ComfyUI directory not found!"
    echo "Please ensure ComfyUI is installed in /workspaces/SH-Learning-AI/ComfyUI"
    exit 1
fi

cd /workspaces/SH-Learning-AI/ComfyUI

# Check for models
if [ ! -d "models/checkpoints" ] || [ -z "$(ls -A models/checkpoints 2>/dev/null)" ]; then
    echo "⚠️  Warning: No models found in ComfyUI/models/checkpoints/"
    echo ""
    echo "To download a model, run one of these commands:"
    echo ""
    echo "Option 1 - SDXL (best quality, 6.5GB):"
    echo "  cd /workspaces/SH-Learning-AI/ComfyUI/models/checkpoints"
    echo "  wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    echo ""
    echo "Option 2 - SD 1.5 (faster, 2GB):"
    echo "  cd /workspaces/SH-Learning-AI/ComfyUI/models/checkpoints"
    echo "  wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"
    echo ""
    echo "Press Enter to continue anyway, or Ctrl+C to exit..."
    read
fi

echo "✅ Starting ComfyUI on port 8188..."
echo "   Access the UI at: http://localhost:8188"
echo ""
echo "Voice commands will use the API automatically."
echo "Press Ctrl+C to stop the server."
echo "================================"
echo ""

# Start ComfyUI with API enabled
python main.py --listen --port 8188