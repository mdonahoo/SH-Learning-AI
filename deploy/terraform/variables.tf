# ==============================================================================
# Starship Horizons Learning AI - Terraform Variables
# ==============================================================================

# ------------------------------------------------------------------------------
# Cloud Provider Selection
# ------------------------------------------------------------------------------

variable "cloud_provider" {
  description = "Cloud provider to deploy to (azure, aws, gcp)"
  type        = string
  default     = "azure"

  validation {
    condition     = contains(["azure", "aws", "gcp"], var.cloud_provider)
    error_message = "Cloud provider must be one of: azure, aws, gcp"
  }
}

# ------------------------------------------------------------------------------
# Common Configuration
# ------------------------------------------------------------------------------

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "sh-learning-ai"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vm_name" {
  description = "Name for the VM instance (also used for DNS)"
  type        = string
  default     = "sh-learning-ai-vm"
}

variable "region" {
  description = "Cloud region to deploy to"
  type        = string
  default     = "eastus"  # Azure default; AWS: us-east-1, GCP: us-east1
}

# ------------------------------------------------------------------------------
# VM Configuration
# ------------------------------------------------------------------------------

variable "vm_size" {
  description = "VM size/instance type (cloud-specific)"
  type        = string
  default     = ""  # Will use cloud-specific defaults
}

variable "os_disk_size_gb" {
  description = "OS disk size in GB"
  type        = number
  default     = 128
}

variable "admin_username" {
  description = "Admin username for SSH access"
  type        = string
  default     = "azureuser"
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key file"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "ssh_public_key" {
  description = "SSH public key content (alternative to path)"
  type        = string
  default     = ""
  sensitive   = true
}

# ------------------------------------------------------------------------------
# Application Configuration
# ------------------------------------------------------------------------------

variable "git_repo_url" {
  description = "Git repository URL to clone"
  type        = string
  default     = ""
}

variable "whisper_model" {
  description = "Whisper model size (tiny, base, small, medium, large-v2, large-v3)"
  type        = string
  default     = "medium"

  validation {
    condition     = contains(["tiny", "base", "small", "medium", "large-v2", "large-v3"], var.whisper_model)
    error_message = "Whisper model must be one of: tiny, base, small, medium, large-v2, large-v3"
  }
}

variable "ollama_model" {
  description = "Ollama LLM model name"
  type        = string
  default     = "llama3.2"
}

variable "web_server_port" {
  description = "Port for the web server"
  type        = number
  default     = 8000
}

# ------------------------------------------------------------------------------
# Security Configuration
# ------------------------------------------------------------------------------

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "allowed_web_cidrs" {
  description = "CIDR blocks allowed for web access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ------------------------------------------------------------------------------
# Cloud-Specific Overrides
# ------------------------------------------------------------------------------

# Azure-specific
variable "azure_subscription_id" {
  description = "Azure subscription ID"
  type        = string
  default     = ""
}

variable "azure_resource_group_name" {
  description = "Azure resource group name (created if not exists)"
  type        = string
  default     = ""
}

# AWS-specific
variable "aws_profile" {
  description = "AWS CLI profile to use"
  type        = string
  default     = "default"
}

variable "aws_vpc_id" {
  description = "Existing VPC ID (creates new if empty)"
  type        = string
  default     = ""
}

# GCP-specific
variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = ""
}

variable "gcp_zone" {
  description = "GCP zone (defaults to region-a)"
  type        = string
  default     = ""
}

# ------------------------------------------------------------------------------
# Tags
# ------------------------------------------------------------------------------

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default     = {}
}
