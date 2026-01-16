# Terraform Deployment (Cloud-Agnostic)

Deploy Starship Horizons Learning AI to **Azure**, **AWS**, or **GCP** using Terraform.

## Quick Start

### 1. Initialize Terraform

```bash
cd deploy/terraform
terraform init
```

### 2. Deploy to Your Cloud

**Azure:**
```bash
terraform apply -var-file=examples/azure.tfvars \
  -var="vm_name=vm-alpha" \
  -var="git_repo_url=https://github.com/YOUR_USER/SH-Learning-AI.git"
```

**AWS:**
```bash
terraform apply -var-file=examples/aws.tfvars \
  -var="vm_name=vm-alpha" \
  -var="git_repo_url=https://github.com/YOUR_USER/SH-Learning-AI.git"
```

**GCP:**
```bash
terraform apply -var-file=examples/gcp.tfvars \
  -var="vm_name=vm-alpha" \
  -var="gcp_project_id=your-project-id" \
  -var="git_repo_url=https://github.com/YOUR_USER/SH-Learning-AI.git"
```

### 3. After Deployment

```bash
# Get SSH command from outputs
terraform output ssh_command

# SSH into the VM
ssh azureuser@vm-alpha.eastus.cloudapp.azure.com

# Start the application
~/start-app.sh

# Access the web interface
terraform output web_url
```

## Deploy Multiple VMs

Deploy multiple VMs with different names using workspaces:

```bash
# Create and deploy vm-alpha
terraform workspace new vm-alpha
terraform apply -var-file=examples/azure.tfvars -var="vm_name=vm-alpha"

# Create and deploy vm-beta
terraform workspace new vm-beta
terraform apply -var-file=examples/azure.tfvars -var="vm_name=vm-beta"

# Create and deploy vm-gamma
terraform workspace new vm-gamma
terraform apply -var-file=examples/azure.tfvars -var="vm_name=vm-gamma"

# List all workspaces
terraform workspace list

# Switch between workspaces
terraform workspace select vm-alpha
```

Each VM gets its own DNS name:
- `vm-alpha.eastus.cloudapp.azure.com`
- `vm-beta.eastus.cloudapp.azure.com`
- `vm-gamma.eastus.cloudapp.azure.com`

## GPU Instance Comparison

| Cloud | Instance Type | GPU | vCPUs | RAM | ~Hourly Cost |
|-------|--------------|-----|-------|-----|--------------|
| **Azure** | Standard_NC4as_T4_v3 | T4 16GB | 4 | 28GB | $0.53 |
| **AWS** | g4dn.xlarge | T4 16GB | 4 | 16GB | $0.53 |
| **GCP** | n1-standard-4 + T4 | T4 16GB | 4 | 15GB | $0.55 |

## Configuration Variables

### Common Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `cloud_provider` | azure | Cloud to deploy to (azure, aws, gcp) |
| `vm_name` | sh-learning-ai-vm | VM name (used for DNS) |
| `region` | eastus | Cloud region |
| `admin_username` | azureuser | SSH username |
| `ssh_public_key_path` | ~/.ssh/id_rsa.pub | Path to SSH public key |
| `git_repo_url` | "" | Git repo to clone |
| `whisper_model` | medium | Whisper model size |
| `ollama_model` | llama3.2 | Ollama LLM model |
| `web_server_port` | 8000 | Web server port |

### Security Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `allowed_ssh_cidrs` | ["0.0.0.0/0"] | CIDRs allowed for SSH |
| `allowed_web_cidrs` | ["0.0.0.0/0"] | CIDRs allowed for web access |

**Production tip:** Restrict to your IP:
```bash
terraform apply -var='allowed_ssh_cidrs=["YOUR_IP/32"]'
```

### Cloud-Specific Variables

**Azure:**
| Variable | Description |
|----------|-------------|
| `azure_subscription_id` | Azure subscription ID |
| `azure_resource_group_name` | Custom resource group name |

**AWS:**
| Variable | Description |
|----------|-------------|
| `aws_profile` | AWS CLI profile |
| `aws_vpc_id` | Existing VPC ID (creates new if empty) |

**GCP:**
| Variable | Description |
|----------|-------------|
| `gcp_project_id` | GCP project ID (required) |
| `gcp_zone` | GCP zone (defaults to region-a) |

## File Structure

```
deploy/terraform/
├── main.tf              # Main configuration with provider selection
├── variables.tf         # Variable definitions
├── outputs.tf           # Output definitions
├── templates/
│   └── cloud-init.yaml.tpl  # Cloud-init template
├── modules/
│   ├── azure/           # Azure-specific resources
│   ├── aws/             # AWS-specific resources
│   └── gcp/             # GCP-specific resources
├── examples/
│   ├── azure.tfvars     # Azure example config
│   ├── aws.tfvars       # AWS example config
│   └── gcp.tfvars       # GCP example config
└── README.md            # This file
```

## Common Operations

### View Outputs

```bash
terraform output
terraform output web_url
terraform output ssh_command
```

### Destroy Resources

```bash
# Destroy current workspace
terraform destroy -var-file=examples/azure.tfvars

# Destroy specific workspace
terraform workspace select vm-alpha
terraform destroy -var-file=examples/azure.tfvars
```

### Update Configuration

```bash
# Change Ollama model
terraform apply -var-file=examples/azure.tfvars -var="ollama_model=qwen2.5:7b"

# Change Whisper model
terraform apply -var-file=examples/azure.tfvars -var="whisper_model=large-v3"
```

## Prerequisites

### Azure
```bash
az login
az account set --subscription "Your Subscription"
```

### AWS
```bash
aws configure
# or use AWS_PROFILE environment variable
```

### GCP
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

## Troubleshooting

### Terraform init fails
```bash
# Clear cache and reinitialize
rm -rf .terraform .terraform.lock.hcl
terraform init
```

### GPU quota issues
All clouds require GPU quota approval:
- **Azure:** Request NC-series quota in portal
- **AWS:** Request g4dn limit increase
- **GCP:** Request GPU quota in console

### State conflicts with workspaces
```bash
# List workspaces
terraform workspace list

# Delete workspace (after destroying resources)
terraform workspace select default
terraform workspace delete vm-old
```

## Cost Management

### Stop VM (all clouds)
VMs can be stopped to save costs when not in use:

**Azure:**
```bash
az vm deallocate -g sh-learning-ai-dev-rg -n vm-alpha
az vm start -g sh-learning-ai-dev-rg -n vm-alpha
```

**AWS:**
```bash
aws ec2 stop-instances --instance-ids $(terraform output -raw aws_instance_id)
aws ec2 start-instances --instance-ids $(terraform output -raw aws_instance_id)
```

**GCP:**
```bash
gcloud compute instances stop vm-alpha --zone=$(terraform output -raw gcp_zone)
gcloud compute instances start vm-alpha --zone=$(terraform output -raw gcp_zone)
```

### Destroy all resources
```bash
terraform destroy -var-file=examples/azure.tfvars
```
