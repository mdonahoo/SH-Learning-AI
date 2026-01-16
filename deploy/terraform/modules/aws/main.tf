# ==============================================================================
# AWS Module for Starship Horizons Learning AI
# ==============================================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------

variable "project_name" { type = string }
variable "environment" { type = string }
variable "vm_name" { type = string }
variable "region" { type = string }
variable "vpc_id" { type = string }
variable "instance_type" { type = string }
variable "os_disk_size_gb" { type = number }
variable "admin_username" { type = string }
variable "ssh_public_key" { type = string }
variable "cloud_init" { type = string }
variable "web_server_port" { type = number }
variable "allowed_ssh_cidrs" { type = list(string) }
variable "allowed_web_cidrs" { type = list(string) }
variable "tags" { type = map(string) }

# ------------------------------------------------------------------------------
# Data Sources
# ------------------------------------------------------------------------------

# Get latest Ubuntu 22.04 AMI with GPU support
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Get availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Get existing VPC or use default
data "aws_vpc" "selected" {
  count = var.vpc_id != "" ? 1 : 0
  id    = var.vpc_id
}

# ------------------------------------------------------------------------------
# VPC (create if not provided)
# ------------------------------------------------------------------------------

resource "aws_vpc" "main" {
  count                = var.vpc_id == "" ? 1 : 0
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(var.tags, {
    Name = "${var.vm_name}-vpc"
  })
}

locals {
  vpc_id = var.vpc_id != "" ? var.vpc_id : aws_vpc.main[0].id
}

resource "aws_internet_gateway" "main" {
  count  = var.vpc_id == "" ? 1 : 0
  vpc_id = local.vpc_id

  tags = merge(var.tags, {
    Name = "${var.vm_name}-igw"
  })
}

resource "aws_subnet" "main" {
  count                   = var.vpc_id == "" ? 1 : 0
  vpc_id                  = local.vpc_id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name = "${var.vm_name}-subnet"
  })
}

resource "aws_route_table" "main" {
  count  = var.vpc_id == "" ? 1 : 0
  vpc_id = local.vpc_id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main[0].id
  }

  tags = merge(var.tags, {
    Name = "${var.vm_name}-rt"
  })
}

resource "aws_route_table_association" "main" {
  count          = var.vpc_id == "" ? 1 : 0
  subnet_id      = aws_subnet.main[0].id
  route_table_id = aws_route_table.main[0].id
}

# Get existing subnets if VPC provided
data "aws_subnets" "existing" {
  count = var.vpc_id != "" ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }
}

locals {
  subnet_id = var.vpc_id != "" ? data.aws_subnets.existing[0].ids[0] : aws_subnet.main[0].id
}

# ------------------------------------------------------------------------------
# Security Group
# ------------------------------------------------------------------------------

resource "aws_security_group" "main" {
  name        = "${var.vm_name}-sg"
  description = "Security group for ${var.vm_name}"
  vpc_id      = local.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_ssh_cidrs
  }

  ingress {
    description = "Web Server"
    from_port   = var.web_server_port
    to_port     = var.web_server_port
    protocol    = "tcp"
    cidr_blocks = var.allowed_web_cidrs
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_web_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.vm_name}-sg"
  })
}

# ------------------------------------------------------------------------------
# SSH Key Pair
# ------------------------------------------------------------------------------

resource "aws_key_pair" "main" {
  key_name   = "${var.vm_name}-key"
  public_key = var.ssh_public_key

  tags = var.tags
}

# ------------------------------------------------------------------------------
# EC2 Instance
# ------------------------------------------------------------------------------

resource "aws_instance" "main" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = var.instance_type
  subnet_id     = local.subnet_id
  key_name      = aws_key_pair.main.key_name

  vpc_security_group_ids = [aws_security_group.main.id]

  root_block_device {
    volume_size           = var.os_disk_size_gb
    volume_type           = "gp3"
    delete_on_termination = true
  }

  user_data = var.cloud_init

  tags = merge(var.tags, {
    Name = var.vm_name
  })

  # GPU instances may take longer to initialize
  timeouts {
    create = "30m"
  }
}

# ------------------------------------------------------------------------------
# Elastic IP (for stable public IP)
# ------------------------------------------------------------------------------

resource "aws_eip" "main" {
  instance = aws_instance.main.id
  domain   = "vpc"

  tags = merge(var.tags, {
    Name = "${var.vm_name}-eip"
  })
}

# ------------------------------------------------------------------------------
# Outputs
# ------------------------------------------------------------------------------

output "public_ip" {
  value = aws_eip.main.public_ip
}

output "public_dns" {
  value = aws_eip.main.public_dns
}

output "ssh_command" {
  value = "ssh ${var.admin_username}@${aws_eip.main.public_ip}"
}

output "web_url" {
  value = "http://${aws_eip.main.public_ip}:${var.web_server_port}"
}

output "instance_id" {
  value = aws_instance.main.id
}
