FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    curl wget git sudo \
    pulseaudio-utils alsa-utils \
    portaudio19-dev libsndfile1 \
    coreutils util-linux procps \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# Install PyTorch with CUDA first
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other packages from PyPI
RUN pip install --no-cache-dir \
    openai-whisper sounddevice rich streamlit \
    SpeechRecognition pygame edge-tts pyttsx3

RUN useradd -m -s /bin/bash -G sudo vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    usermod -a -G audio vscode && \
    chown -R vscode:vscode /opt/venv


WORKDIR /workspaces/SH-Learning-AI

# Ensure the workspace directory exists and has correct permissions
RUN mkdir -p /workspaces/SH-Learning-AI && \
    chown -R vscode:vscode /workspaces

# Fix permissions for the entire workspace on container start
RUN echo '#!/bin/bash\nif [ -d "/workspaces/SH-Learning-AI" ]; then\n    sudo chown -R vscode:vscode /workspaces/SH-Learning-AI\n    sudo find /workspaces/SH-Learning-AI -type d -exec chmod 755 {} \\;\n    sudo find /workspaces/SH-Learning-AI -type f -exec chmod 644 {} \\;\nfi' > /usr/local/bin/fix-permissions.sh && \
    chmod +x /usr/local/bin/fix-permissions.sh

USER vscode

# Run permission fix script and keep container running
CMD ["/bin/bash", "-c", "/usr/local/bin/fix-permissions.sh && while true; do /bin/sleep 30; done"]