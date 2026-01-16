# ==============================================================================
# GCP Deployment Example
# ==============================================================================
# Usage:
#   terraform init
#   terraform plan -var-file=examples/gcp.tfvars
#   terraform apply -var-file=examples/gcp.tfvars
# ==============================================================================

cloud_provider = "gcp"

# VM Configuration
vm_name     = "vm-alpha"
region      = "us-east1"
vm_size     = "n1-standard-4"      # 4 vCPU, 15GB RAM + T4 GPU attached
os_disk_size_gb = 128

# Alternative machine types:
# n1-standard-4  - 4 vCPU, 15GB RAM (+ GPU)
# n1-standard-8  - 8 vCPU, 30GB RAM (+ GPU)
# a2-highgpu-1g  - 12 vCPU, 85GB RAM, 1x A100 (expensive but fast)

# Authentication
admin_username      = "gcpuser"
ssh_public_key_path = "~/.ssh/id_rsa.pub"

# Application
git_repo_url   = ""                # Set to your repo URL
whisper_model  = "medium"
ollama_model   = "llama3.2"
web_server_port = 8000

# Security (restrict for production)
allowed_ssh_cidrs = ["0.0.0.0/0"]
allowed_web_cidrs = ["0.0.0.0/0"]

# GCP-specific (required)
gcp_project_id = "your-gcp-project-id"   # REQUIRED: Set your GCP project ID
# gcp_zone      = "us-east1-b"           # Optional: defaults to region-a
