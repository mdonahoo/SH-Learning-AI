# ==============================================================================
# Azure Deployment Example
# ==============================================================================
# Usage:
#   terraform init
#   terraform plan -var-file=examples/azure.tfvars
#   terraform apply -var-file=examples/azure.tfvars
# ==============================================================================

cloud_provider = "azure"

# VM Configuration
vm_name     = "vm-alpha"           # Creates: vm-alpha.eastus.cloudapp.azure.com
region      = "eastus"
vm_size     = "Standard_NC4as_T4_v3"  # NVIDIA T4 GPU
os_disk_size_gb = 128

# Authentication
admin_username      = "azureuser"
ssh_public_key_path = "~/.ssh/id_rsa.pub"

# Application
git_repo_url   = ""                # Set to your repo URL
whisper_model  = "medium"
ollama_model   = "llama3.2"
web_server_port = 8000

# Security (restrict for production)
allowed_ssh_cidrs = ["0.0.0.0/0"]   # Restrict to your IP in production
allowed_web_cidrs = ["0.0.0.0/0"]

# Tags
tags = {
  Owner = "your-name"
  Team  = "your-team"
}

# Azure-specific (optional)
# azure_subscription_id    = "your-subscription-id"
# azure_resource_group_name = "my-custom-rg"
