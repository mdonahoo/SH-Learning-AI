# ==============================================================================
# Outputs
# ==============================================================================

# ------------------------------------------------------------------------------
# Azure Outputs
# ------------------------------------------------------------------------------

output "azure_public_ip" {
  description = "Azure VM public IP address"
  value       = var.cloud_provider == "azure" ? module.azure[0].public_ip : null
}

output "azure_fqdn" {
  description = "Azure VM fully qualified domain name"
  value       = var.cloud_provider == "azure" ? module.azure[0].fqdn : null
}

output "azure_ssh_command" {
  description = "SSH command for Azure VM"
  value       = var.cloud_provider == "azure" ? module.azure[0].ssh_command : null
}

output "azure_web_url" {
  description = "Web URL for Azure VM"
  value       = var.cloud_provider == "azure" ? module.azure[0].web_url : null
}

# ------------------------------------------------------------------------------
# AWS Outputs
# ------------------------------------------------------------------------------

output "aws_public_ip" {
  description = "AWS EC2 instance public IP address"
  value       = var.cloud_provider == "aws" ? module.aws[0].public_ip : null
}

output "aws_public_dns" {
  description = "AWS EC2 instance public DNS"
  value       = var.cloud_provider == "aws" ? module.aws[0].public_dns : null
}

output "aws_ssh_command" {
  description = "SSH command for AWS EC2 instance"
  value       = var.cloud_provider == "aws" ? module.aws[0].ssh_command : null
}

output "aws_web_url" {
  description = "Web URL for AWS EC2 instance"
  value       = var.cloud_provider == "aws" ? module.aws[0].web_url : null
}

# ------------------------------------------------------------------------------
# GCP Outputs
# ------------------------------------------------------------------------------

output "gcp_public_ip" {
  description = "GCP VM public IP address"
  value       = var.cloud_provider == "gcp" ? module.gcp[0].public_ip : null
}

output "gcp_ssh_command" {
  description = "SSH command for GCP VM"
  value       = var.cloud_provider == "gcp" ? module.gcp[0].ssh_command : null
}

output "gcp_web_url" {
  description = "Web URL for GCP VM"
  value       = var.cloud_provider == "gcp" ? module.gcp[0].web_url : null
}

# ------------------------------------------------------------------------------
# Common Outputs (cloud-agnostic)
# ------------------------------------------------------------------------------

output "cloud_provider" {
  description = "Selected cloud provider"
  value       = var.cloud_provider
}

output "vm_name" {
  description = "VM instance name"
  value       = var.vm_name
}

output "public_ip" {
  description = "Public IP address (cloud-agnostic)"
  value = (
    var.cloud_provider == "azure" ? module.azure[0].public_ip :
    var.cloud_provider == "aws" ? module.aws[0].public_ip :
    var.cloud_provider == "gcp" ? module.gcp[0].public_ip :
    null
  )
}

output "ssh_command" {
  description = "SSH command (cloud-agnostic)"
  value = (
    var.cloud_provider == "azure" ? module.azure[0].ssh_command :
    var.cloud_provider == "aws" ? module.aws[0].ssh_command :
    var.cloud_provider == "gcp" ? module.gcp[0].ssh_command :
    null
  )
}

output "web_url" {
  description = "Web URL (cloud-agnostic)"
  value = (
    var.cloud_provider == "azure" ? module.azure[0].web_url :
    var.cloud_provider == "aws" ? module.aws[0].web_url :
    var.cloud_provider == "gcp" ? module.gcp[0].web_url :
    null
  )
}

output "next_steps" {
  description = "Instructions for next steps"
  value       = <<-EOT

    ===========================================
    Deployment Complete!
    ===========================================

    1. Wait 5-10 minutes for cloud-init to complete

    2. SSH into the VM:
       ${var.cloud_provider == "azure" ? module.azure[0].ssh_command : var.cloud_provider == "aws" ? module.aws[0].ssh_command : module.gcp[0].ssh_command}

    3. Check cloud-init status:
       cloud-init status --wait

    4. Start the application:
       ~/start-app.sh

    5. Access the web interface:
       ${var.cloud_provider == "azure" ? module.azure[0].web_url : var.cloud_provider == "aws" ? module.aws[0].web_url : module.gcp[0].web_url}

    ===========================================
  EOT
}
