# ==============================================================================
# GCP Module for Starship Horizons Learning AI
# ==============================================================================

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

variable "project_name" { type = string }
variable "project_id" { type = string }
variable "environment" { type = string }
variable "vm_name" { type = string }
variable "region" { type = string }
variable "zone" { type = string }
variable "machine_type" { type = string }
variable "os_disk_size_gb" { type = number }
variable "admin_username" { type = string }
variable "ssh_public_key" { type = string }
variable "cloud_init" { type = string }
variable "web_server_port" { type = number }
variable "allowed_ssh_cidrs" { type = list(string) }
variable "allowed_web_cidrs" { type = list(string) }

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

resource "google_compute_network" "main" {
  name                    = "${var.vm_name}-network"
  project                 = var.project_id
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "main" {
  name          = "${var.vm_name}-subnet"
  project       = var.project_id
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.main.id
}

# ------------------------------------------------------------------------------
# Firewall Rules
# ------------------------------------------------------------------------------

resource "google_compute_firewall" "ssh" {
  name    = "${var.vm_name}-allow-ssh"
  project = var.project_id
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.allowed_ssh_cidrs
  target_tags   = ["${var.vm_name}-server"]
}

resource "google_compute_firewall" "web" {
  name    = "${var.vm_name}-allow-web"
  project = var.project_id
  network = google_compute_network.main.name

  allow {
    protocol = "tcp"
    ports    = [tostring(var.web_server_port), "443"]
  }

  source_ranges = var.allowed_web_cidrs
  target_tags   = ["${var.vm_name}-server"]
}

# ------------------------------------------------------------------------------
# External IP
# ------------------------------------------------------------------------------

resource "google_compute_address" "main" {
  name    = "${var.vm_name}-ip"
  project = var.project_id
  region  = var.region
}

# ------------------------------------------------------------------------------
# Compute Instance
# ------------------------------------------------------------------------------

resource "google_compute_instance" "main" {
  name         = var.vm_name
  project      = var.project_id
  machine_type = var.machine_type
  zone         = var.zone

  tags = ["${var.vm_name}-server"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = var.os_disk_size_gb
      type  = "pd-ssd"
    }
  }

  # NVIDIA T4 GPU
  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  # Required for GPU instances
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  network_interface {
    subnetwork = google_compute_subnetwork.main.id

    access_config {
      nat_ip = google_compute_address.main.address
    }
  }

  metadata = {
    ssh-keys  = "${var.admin_username}:${var.ssh_public_key}"
    user-data = var.cloud_init
  }

  # Enable startup script via cloud-init
  metadata_startup_script = <<-EOF
    #!/bin/bash
    # Cloud-init should handle most setup, this is a fallback
    if ! command -v docker &> /dev/null; then
      curl -fsSL https://get.docker.com | sh
    fi
  EOF

  labels = {
    project     = lower(replace(var.project_name, "/[^a-z0-9-]/", "-"))
    environment = var.environment
    managed-by  = "terraform"
  }

  # Allow stopping for updates
  allow_stopping_for_update = true
}

# ------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------

output "public_ip" {
  value = google_compute_address.main.address
}

output "ssh_command" {
  value = "ssh ${var.admin_username}@${google_compute_address.main.address}"
}

output "web_url" {
  value = "http://${google_compute_address.main.address}:${var.web_server_port}"
}

output "instance_name" {
  value = google_compute_instance.main.name
}

output "zone" {
  value = google_compute_instance.main.zone
}
