# ==============================================================================
# AWS Deployment Example
# ==============================================================================
# Usage:
#   terraform init
#   terraform plan -var-file=examples/aws.tfvars
#   terraform apply -var-file=examples/aws.tfvars
# ==============================================================================

cloud_provider = "aws"

# VM Configuration
vm_name     = "vm-alpha"
region      = "us-east-1"
vm_size     = "g4dn.xlarge"        # NVIDIA T4 GPU (4 vCPU, 16GB RAM, 16GB GPU)
os_disk_size_gb = 128

# Alternative GPU instance types:
# g4dn.xlarge  - 1x T4, 4 vCPU, 16GB RAM (~$0.53/hr)
# g4dn.2xlarge - 1x T4, 8 vCPU, 32GB RAM (~$0.75/hr)
# g5.xlarge    - 1x A10G, 4 vCPU, 16GB RAM (~$1.00/hr) - faster

# Authentication
admin_username      = "ubuntu"     # Ubuntu default user
ssh_public_key_path = "~/.ssh/id_rsa.pub"

# Application
git_repo_url   = ""                # Set to your repo URL
whisper_model  = "large-v3"
ollama_model   = "llama3.2"
web_server_port = 8000

# Security (restrict for production)
allowed_ssh_cidrs = ["0.0.0.0/0"]
allowed_web_cidrs = ["0.0.0.0/0"]

# Tags
tags = {
  Owner = "your-name"
  Team  = "your-team"
}

# AWS-specific (optional)
# aws_profile = "default"
# aws_vpc_id  = "vpc-xxxxxxxx"     # Use existing VPC
