#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Check if .env file exists
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Creating .env file..."
    cat > "$SCRIPT_DIR/.env" << EOL
ANTHROPIC_API_KEY=your_api_key_here
EOL
    echo "Please edit .env file and add your Anthropic API key"
fi

# Check if conda environment 'pfa' exists
if ! conda info --envs | grep -q "^pfa "; then
    echo "Creating new conda environment 'pfa'..."
    conda create -n pfa python=3.10 -y
else
    echo "Conda environment 'pfa' already exists"
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate pfa

# Create requirements.txt if it doesn't exist
if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Creating requirements.txt..."
    cat > "$SCRIPT_DIR/requirements.txt" << EOL
anthropic
pypdf
pandas
python-dotenv
EOL
fi

# Install requirements
echo "Installing requirements..."
pip install -r "$SCRIPT_DIR/requirements.txt"

# Create data directory
mkdir -p "$SCRIPT_DIR/../../data/pfa"

echo "Initialization complete! Environment 'pfa' is ready."
