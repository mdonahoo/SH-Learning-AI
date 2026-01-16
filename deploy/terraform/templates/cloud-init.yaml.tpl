#cloud-config
# ==============================================================================
# Cloud-init configuration for Starship Horizons Learning AI
# Works across Azure, AWS, and GCP
# ==============================================================================

package_update: true
package_upgrade: true

packages:
  - apt-transport-https
  - ca-certificates
  - curl
  - gnupg
  - lsb-release
  - git
  - ffmpeg
  - python3-pip
  - jq
  - unzip

write_files:
  - path: /home/${admin_username}/start-app.sh
    permissions: "0755"
    owner: ${admin_username}:${admin_username}
    content: |
      #!/bin/bash
      set -e
      echo "=== Starting Starship Horizons Learning AI ==="

      cd /home/${admin_username}/app

      # Build Docker image
      echo "Building Docker image..."
      docker build -t sh-learning-ai .

      # Stop existing container if running
      docker stop sh-learning-ai 2>/dev/null || true
      docker rm sh-learning-ai 2>/dev/null || true

      # Run container with GPU support
      echo "Starting container..."
      docker run -d \
        --gpus all \
        -p ${web_server_port}:${web_server_port} \
        --name sh-learning-ai \
        -e WHISPER_MODEL=${whisper_model} \
        -e OLLAMA_MODEL=${ollama_model} \
        -e WEB_SERVER_PORT=${web_server_port} \
        -v /home/${admin_username}/ollama-models:/root/.ollama \
        --restart unless-stopped \
        sh-learning-ai

      echo "=== Application started ==="
      echo "Web interface: http://$(curl -s ifconfig.me):${web_server_port}"
      echo "View logs: docker logs -f sh-learning-ai"

  - path: /home/${admin_username}/check-status.sh
    permissions: "0755"
    owner: ${admin_username}:${admin_username}
    content: |
      #!/bin/bash
      echo "=== System Status ==="
      echo ""
      echo "GPU Status:"
      nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null || echo "GPU not available yet"
      echo ""
      echo "Docker Containers:"
      docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || echo "Docker not ready"
      echo ""
      echo "Application Health:"
      curl -s http://localhost:${web_server_port}/api/health | jq . 2>/dev/null || echo "Application not responding"

  - path: /home/${admin_username}/install-gpu-drivers.sh
    permissions: "0755"
    owner: root:root
    content: |
      #!/bin/bash
      set -e
      echo "Installing NVIDIA drivers and container toolkit..."

      # Detect cloud provider and install appropriate drivers
      if curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/ > /dev/null 2>&1; then
        echo "Detected GCP - using NVIDIA installer"
        curl -fsSL https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py -o /tmp/install_gpu_driver.py
        python3 /tmp/install_gpu_driver.py
      elif curl -s http://169.254.169.254/latest/meta-data/ > /dev/null 2>&1; then
        echo "Detected AWS - installing NVIDIA drivers"
        apt-get install -y linux-headers-$(uname -r)
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed 's/\.//')
        wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
        dpkg -i cuda-keyring_1.0-1_all.deb
        apt-get update
        apt-get install -y nvidia-driver-535
      else
        echo "Detected Azure - installing NVIDIA drivers"
        apt-get install -y nvidia-driver-535
      fi

      # Install NVIDIA container toolkit (works for all clouds)
      distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
      apt-get update
      apt-get install -y nvidia-container-toolkit

      # Configure Docker to use NVIDIA runtime
      nvidia-ctk runtime configure --runtime=docker
      systemctl restart docker

      echo "NVIDIA setup complete!"

runcmd:
  # Install Docker
  - curl -fsSL https://get.docker.com | sh
  - systemctl enable docker
  - systemctl start docker
  - usermod -aG docker ${admin_username}

  # Install GPU drivers (cloud-agnostic script)
  - /home/${admin_username}/install-gpu-drivers.sh || echo "GPU driver installation may require reboot"

  # Create Ollama models directory
  - mkdir -p /home/${admin_username}/ollama-models
  - chown ${admin_username}:${admin_username} /home/${admin_username}/ollama-models

  # Clone repository if specified
%{ if git_repo_url != "" ~}
  - cd /home/${admin_username} && git clone ${git_repo_url} app && chown -R ${admin_username}:${admin_username} app
%{ else ~}
  - echo "No git repo specified - clone manually to /home/${admin_username}/app"
%{ endif ~}

final_message: |
  ===========================================
  Cloud-init completed!
  ===========================================

  Next steps:
  1. SSH into the VM
  2. Run: ~/start-app.sh
  3. Wait for Docker build and model downloads
  4. Access: http://YOUR_IP:${web_server_port}

  Useful commands:
  - ~/check-status.sh        - Check system status
  - docker logs -f sh-learning-ai - View logs
  ===========================================
