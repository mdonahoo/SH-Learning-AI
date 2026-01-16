# ==============================================================================
# Starship Horizons Learning AI - Main Terraform Configuration
#
# Deploy to Azure, AWS, or GCP by setting the cloud_provider variable
# ==============================================================================

terraform {
  required_version = ">= 1.0.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ------------------------------------------------------------------------------
# Local Variables
# ------------------------------------------------------------------------------

locals {
  # Merge default tags with user-provided tags
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    },
    var.tags
  )

  # Read SSH key from file if not provided directly
  ssh_public_key = var.ssh_public_key != "" ? var.ssh_public_key : file(pathexpand(var.ssh_public_key_path))

  # Cloud-init template
  cloud_init = templatefile("${path.module}/templates/cloud-init.yaml.tpl", {
    admin_username  = var.admin_username
    git_repo_url    = var.git_repo_url
    whisper_model   = var.whisper_model
    ollama_model    = var.ollama_model
    web_server_port = var.web_server_port
  })

  # Region mapping for multi-cloud
  region_map = {
    # US East
    "eastus"    = { azure = "eastus", aws = "us-east-1", gcp = "us-east1" }
    "us-east-1" = { azure = "eastus", aws = "us-east-1", gcp = "us-east1" }
    "us-east1"  = { azure = "eastus", aws = "us-east-1", gcp = "us-east1" }
    # US West
    "westus2"   = { azure = "westus2", aws = "us-west-2", gcp = "us-west1" }
    "us-west-2" = { azure = "westus2", aws = "us-west-2", gcp = "us-west1" }
    "us-west1"  = { azure = "westus2", aws = "us-west-2", gcp = "us-west1" }
    # Europe
    "westeurope"  = { azure = "westeurope", aws = "eu-west-1", gcp = "europe-west1" }
    "eu-west-1"   = { azure = "westeurope", aws = "eu-west-1", gcp = "europe-west1" }
    "europe-west1" = { azure = "westeurope", aws = "eu-west-1", gcp = "europe-west1" }
  }

  # Normalize region
  normalized_region = lookup(local.region_map, var.region, {
    azure = var.region
    aws   = var.region
    gcp   = var.region
  })
}

# ------------------------------------------------------------------------------
# Provider Configuration
# ------------------------------------------------------------------------------

provider "azurerm" {
  features {}
  subscription_id = var.azure_subscription_id != "" ? var.azure_subscription_id : null
  skip_provider_registration = true
}

provider "aws" {
  region  = local.normalized_region.aws
  profile = var.aws_profile
}

provider "google" {
  project = var.gcp_project_id
  region  = local.normalized_region.gcp
}

# ------------------------------------------------------------------------------
# Azure Deployment
# ------------------------------------------------------------------------------

module "azure" {
  source = "./modules/azure"
  count  = var.cloud_provider == "azure" ? 1 : 0

  project_name        = var.project_name
  environment         = var.environment
  vm_name             = var.vm_name
  region              = local.normalized_region.azure
  resource_group_name = var.azure_resource_group_name
  vm_size             = var.vm_size != "" ? var.vm_size : "Standard_NC4as_T4_v3"
  os_disk_size_gb     = var.os_disk_size_gb
  admin_username      = var.admin_username
  ssh_public_key      = local.ssh_public_key
  cloud_init          = local.cloud_init
  web_server_port     = var.web_server_port
  allowed_ssh_cidrs   = var.allowed_ssh_cidrs
  allowed_web_cidrs   = var.allowed_web_cidrs
  tags                = local.common_tags
}

# ------------------------------------------------------------------------------
# AWS Deployment
# ------------------------------------------------------------------------------

module "aws" {
  source = "./modules/aws"
  count  = var.cloud_provider == "aws" ? 1 : 0

  project_name      = var.project_name
  environment       = var.environment
  vm_name           = var.vm_name
  region            = local.normalized_region.aws
  vpc_id            = var.aws_vpc_id
  instance_type     = var.vm_size != "" ? var.vm_size : "g4dn.xlarge"
  os_disk_size_gb   = var.os_disk_size_gb
  admin_username    = var.admin_username
  ssh_public_key    = local.ssh_public_key
  cloud_init        = local.cloud_init
  web_server_port   = var.web_server_port
  allowed_ssh_cidrs = var.allowed_ssh_cidrs
  allowed_web_cidrs = var.allowed_web_cidrs
  tags              = local.common_tags
}

# ------------------------------------------------------------------------------
# GCP Deployment
# ------------------------------------------------------------------------------

module "gcp" {
  source = "./modules/gcp"
  count  = var.cloud_provider == "gcp" ? 1 : 0

  project_name      = var.project_name
  project_id        = var.gcp_project_id
  environment       = var.environment
  vm_name           = var.vm_name
  region            = local.normalized_region.gcp
  zone              = var.gcp_zone != "" ? var.gcp_zone : "${local.normalized_region.gcp}-a"
  machine_type      = var.vm_size != "" ? var.vm_size : "n1-standard-4"
  os_disk_size_gb   = var.os_disk_size_gb
  admin_username    = var.admin_username
  ssh_public_key    = local.ssh_public_key
  cloud_init        = local.cloud_init
  web_server_port   = var.web_server_port
  allowed_ssh_cidrs = var.allowed_ssh_cidrs
  allowed_web_cidrs = var.allowed_web_cidrs
}
