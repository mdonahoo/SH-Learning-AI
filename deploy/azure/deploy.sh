#!/bin/bash
# ==============================================================================
# Azure Deployment Script for Starship Horizons Learning AI
#
# Deploys a GPU-enabled VM with:
# - NVIDIA drivers and Docker
# - Whisper for audio transcription
# - Ollama for LLM-powered reports
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
RESOURCE_GROUP="sh-learning-ai-rg"
LOCATION="eastus"
DEPLOYMENT_NAME="sh-learning-ai-deployment"
TEMPLATE_FILE="azuredeploy.json"
PARAMETERS_FILE="azuredeploy.parameters.json"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Deploy Starship Horizons Learning AI to Azure with GPU support."
    echo ""
    echo "Options:"
    echo "  -g, --resource-group NAME    Resource group name (default: $RESOURCE_GROUP)"
    echo "  -l, --location LOCATION      Azure region (default: $LOCATION)"
    echo "  -v, --vm-name NAME           VM name - also sets DNS label (default: sh-learning-ai-vm)"
    echo "                               e.g., 'vm-alpha' -> vm-alpha.eastus.cloudapp.azure.com"
    echo "  -k, --ssh-key PATH           Path to SSH public key file"
    echo "  -r, --repo URL               Git repository URL to clone"
    echo "  -s, --vm-size SIZE           VM size (default: Standard_NC4as_T4_v3)"
    echo "  -w, --whisper-model MODEL    Whisper model (tiny/base/small/medium/large-v2/large-v3)"
    echo "  -o, --ollama-model MODEL     Ollama model (llama3.2/qwen2.5:7b/qwen2.5:14b-instruct)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Basic deployment"
    echo "  $0 -k ~/.ssh/id_rsa.pub -r https://github.com/user/SH-Learning-AI.git"
    echo ""
    echo "  # Deploy multiple VMs with friendly names"
    echo "  $0 -v vm-alpha -k ~/.ssh/id_rsa.pub -r https://github.com/user/repo.git"
    echo "  $0 -v vm-beta -k ~/.ssh/id_rsa.pub -r https://github.com/user/repo.git"
    echo ""
    echo "  # Full customization"
    echo "  $0 -g my-rg -l westus2 -v bridge-server -k ~/.ssh/id_ed25519.pub \\"
    echo "     -r https://github.com/user/repo.git -w medium -o qwen2.5:7b"
    exit 1
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}==>${NC} $1"
}

# Parse command line arguments
SSH_KEY_PATH=""
GIT_REPO_URL=""
VM_NAME=""
VM_SIZE=""
WHISPER_MODEL_SIZE=""
OLLAMA_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        -v|--vm-name)
            VM_NAME="$2"
            shift 2
            ;;
        -k|--ssh-key)
            SSH_KEY_PATH="$2"
            shift 2
            ;;
        -r|--repo)
            GIT_REPO_URL="$2"
            shift 2
            ;;
        -s|--vm-size)
            VM_SIZE="$2"
            shift 2
            ;;
        -w|--whisper-model)
            WHISPER_MODEL_SIZE="$2"
            shift 2
            ;;
        -o|--ollama-model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Check prerequisites
log_step "Checking prerequisites..."

if ! command -v az &> /dev/null; then
    log_error "Azure CLI is not installed. Please install it first:"
    echo "  https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    log_warn "jq is not installed. Output parsing may be limited."
fi

# Check if logged in
if ! az account show &> /dev/null; then
    log_warn "Not logged in to Azure. Running 'az login'..."
    az login
fi

# Get SSH key
if [ -z "$SSH_KEY_PATH" ]; then
    # Try default locations
    if [ -f "$HOME/.ssh/id_rsa.pub" ]; then
        SSH_KEY_PATH="$HOME/.ssh/id_rsa.pub"
    elif [ -f "$HOME/.ssh/id_ed25519.pub" ]; then
        SSH_KEY_PATH="$HOME/.ssh/id_ed25519.pub"
    else
        log_error "No SSH key found. Please specify with -k option or generate one:"
        echo "  ssh-keygen -t ed25519 -C 'azure-vm'"
        exit 1
    fi
fi

if [ ! -f "$SSH_KEY_PATH" ]; then
    log_error "SSH key file not found: $SSH_KEY_PATH"
    exit 1
fi

SSH_KEY=$(cat "$SSH_KEY_PATH")
log_info "Using SSH key: $SSH_KEY_PATH"

# Set VM name for display and DNS
DISPLAY_VM_NAME="${VM_NAME:-sh-learning-ai-vm}"
EXPECTED_FQDN="${DISPLAY_VM_NAME}.${LOCATION}.cloudapp.azure.com"

# Display deployment configuration
echo ""
echo "=========================================="
echo "Deployment Configuration"
echo "=========================================="
echo "Resource Group:  $RESOURCE_GROUP"
echo "Location:        $LOCATION"
echo "VM Name:         $DISPLAY_VM_NAME"
echo "DNS Name:        $EXPECTED_FQDN"
echo "SSH Key:         $SSH_KEY_PATH"
[ -n "$GIT_REPO_URL" ] && echo "Git Repository:  $GIT_REPO_URL"
[ -n "$VM_SIZE" ] && echo "VM Size:         $VM_SIZE" || echo "VM Size:         Standard_NC4as_T4_v3 (default)"
[ -n "$WHISPER_MODEL_SIZE" ] && echo "Whisper Model:   $WHISPER_MODEL_SIZE" || echo "Whisper Model:   large-v3 (default)"
[ -n "$OLLAMA_MODEL" ] && echo "Ollama Model:    $OLLAMA_MODEL" || echo "Ollama Model:    llama3.2 (default)"
echo "=========================================="
echo ""
echo "This will deploy:"
echo "  - GPU VM with NVIDIA T4"
echo "  - Docker + NVIDIA Container Toolkit"
echo "  - Whisper for audio transcription"
echo "  - Ollama for LLM report generation"
echo ""
echo "Estimated cost: ~\$380/month (can be stopped when not in use)"
echo ""

read -p "Continue with deployment? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log_info "Deployment cancelled."
    exit 0
fi

# Create resource group
log_step "Creating resource group '$RESOURCE_GROUP' in '$LOCATION'..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# Build parameters override
PARAMS="adminPasswordOrKey=$SSH_KEY"
[ -n "$VM_NAME" ] && PARAMS="$PARAMS vmName=$VM_NAME"
[ -n "$GIT_REPO_URL" ] && PARAMS="$PARAMS gitRepoUrl=$GIT_REPO_URL"
[ -n "$VM_SIZE" ] && PARAMS="$PARAMS vmSize=$VM_SIZE"
[ -n "$WHISPER_MODEL_SIZE" ] && PARAMS="$PARAMS whisperModel=$WHISPER_MODEL_SIZE"
[ -n "$OLLAMA_MODEL" ] && PARAMS="$PARAMS ollamaModel=$OLLAMA_MODEL"

# Deploy template
log_step "Deploying ARM template (this may take 5-10 minutes)..."
DEPLOYMENT_OUTPUT=$(az deployment group create \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --template-file "$SCRIPT_DIR/$TEMPLATE_FILE" \
    --parameters "$SCRIPT_DIR/$PARAMETERS_FILE" \
    --parameters $PARAMS \
    --output json)

# Extract outputs
if command -v jq &> /dev/null; then
    VM_IP=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.vmPublicIP.value')
    VM_FQDN=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.vmFQDN.value')
    SSH_CMD=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.sshCommand.value')
    WEB_URL=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.webServerUrl.value')
else
    VM_IP="(install jq to see)"
    VM_FQDN="(install jq to see)"
    SSH_CMD="ssh azureuser@YOUR_VM_IP"
    WEB_URL="http://YOUR_VM_IP:8000"
fi

# Display results
echo ""
echo "=========================================="
echo -e "${GREEN}Deployment Successful!${NC}"
echo "=========================================="
echo ""
echo "VM Public IP:  $VM_IP"
echo "VM FQDN:       $VM_FQDN"
echo "SSH Command:   $SSH_CMD"
echo "Web Server:    $WEB_URL"
echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Wait 5-10 minutes for cloud-init to complete"
echo ""
echo "2. SSH into the VM:"
echo "   $SSH_CMD"
echo ""
echo "3. Check cloud-init status:"
echo "   cloud-init status --wait"
echo ""
echo "4. Build and start the application:"
echo "   ~/start-app.sh"
echo ""
echo "   This will:"
echo "   - Build the Docker image"
echo "   - Start Ollama and download the LLM model"
echo "   - Start the web server"
echo ""
echo "5. Monitor startup (in another terminal):"
echo "   docker logs -f sh-learning-ai"
echo ""
echo "6. Access the web interface:"
echo "   $WEB_URL"
echo ""
echo "=========================================="
echo "Useful Commands"
echo "=========================================="
echo ""
echo "  ~/check-status.sh         - Check system status"
echo "  docker logs -f sh-learning-ai - View application logs"
echo "  nvidia-smi                - Check GPU status"
echo ""
echo "=========================================="
echo "Cost Management"
echo "=========================================="
echo ""
echo "  Stop VM (saves money):"
echo "    az vm deallocate -g $RESOURCE_GROUP -n sh-learning-ai-vm"
echo ""
echo "  Start VM:"
echo "    az vm start -g $RESOURCE_GROUP -n sh-learning-ai-vm"
echo ""
echo "  Delete everything:"
echo "    az group delete -n $RESOURCE_GROUP"
echo ""
echo "=========================================="
log_info "Deployment complete!"
