#!/bin/bash
set -e  # Exit on error

# Configuration variables
REPO_NAME="auditory-expressions"
REPO_URL="https://github.com/Azuremis/auditory-expressions.git"
PYTHON_VERSION="3.10"
PROJECT_DIR="$HOME/$REPO_NAME"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Check if uv is installed, if not install it
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to the PATH for the current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Source the environment file if it exists
    if [ -f "$HOME/.local/bin/env" ]; then
        echo "Sourcing UV environment..."
        source "$HOME/.local/bin/env"
    fi
    
    # Verify uv is now in PATH
    if ! command -v uv &> /dev/null; then
        echo "ERROR: UV installation succeeded but command not found in PATH."
        echo "Please run the following command and then run this script again:"
        echo "    source \$HOME/.local/bin/env"
        exit 1
    fi
fi

# Check if the project exists, if not clone it
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning $REPO_NAME repository from $REPO_URL..."
    git clone "$REPO_URL" "$PROJECT_DIR"
else
    echo "$REPO_NAME repository already exists at $PROJECT_DIR."
fi

# Change to project directory
cd "$PROJECT_DIR"

# Load environment variables from .env file
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE..."
    # This method properly handles special characters and spaces in values
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip comments and empty lines
        [[ $line =~ ^[[:space:]]*$ || $line =~ ^# ]] && continue
        # Remove quotes and export the variable
        eval "export $(echo "$line" | sed -e 's/[[:space:]]*$//' -e 's/#.*$//')"
    done < "$ENV_FILE"
else
    echo "ERROR: .env file not found at $ENV_FILE!"
    echo "Please create an .env file with the following variables:"
    echo "  WANDB_API_KEY=your_wandb_api_key"
    echo "  GIT_USER=your_git_username"
    echo "  GIT_EMAIL=your_git_email"
    exit 1
fi

# Validate required environment variables
if [ -z "$WANDB_API_KEY" ] || [ -z "$GIT_USER" ] || [ -z "$GIT_EMAIL" ]; then
    echo "ERROR: One or more required environment variables are missing in $ENV_FILE"
    echo "Please ensure WANDB_API_KEY, GIT_USER, and GIT_EMAIL are defined."
    exit 1
fi

# Configure Git credentials using values from .env
echo "Setting up Git configuration..."
git config --global user.name "$GIT_USER"
git config --global user.email "$GIT_EMAIL"

# Check if Python version is available, install if needed
if ! uv python list | grep -q "Python $PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION using UV..."
    uv python install $PYTHON_VERSION
fi

# Pin the Python version for this project
echo "Setting Python version for the project..."
uv python pin $PYTHON_VERSION

# Create and sync virtual environment from pyproject.toml
echo "Creating virtual environment and installing dependencies from pyproject.toml..."
if [ -f "uv.lock" ]; then
    echo "Using existing lockfile for reproducible environment..."
    uv sync --frozen
else
    echo "Creating new lockfile from pyproject.toml..."
    uv sync
fi

# Login to wandb using API key from .env
echo "Logging in to Weights & Biases..."
source .venv/bin/activate
wandb login "$WANDB_API_KEY"

echo "Setup complete! The environment is ready and activated."
# Activate the environment so it's immediately available
source $PROJECT_DIR/.venv/bin/activate