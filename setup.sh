#!/bin/bash
# Quick setup script for PixelShield PoC

set -e

echo "======================================================================="
echo "PixelShield Phase 1: Quick Setup"
echo "======================================================================="

# Check Python version
echo "[*] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "[+] Python $python_version"

if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    echo "[+] Virtual environment created"
else
    echo "[+] Virtual environment already exists"
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[*] Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install requirements
echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify PyTorch
echo "[*] Verifying PyTorch installation..."
python3 -c "import torch; print(f'[+] PyTorch {torch.__version__}')"
python3 -c "import torch; print(f'[+] CUDA available: {torch.cuda.is_available()}')"

# Create directories
echo "[*] Creating output directories..."
mkdir -p images/input
mkdir -p output/cloaked
mkdir -p output/comparison

echo ""
echo "======================================================================="
echo "✓ Setup Complete!"
echo "======================================================================="
echo ""
echo "Next steps:"
echo "  1. Place your test face photo in: images/input/test_face.jpg"
echo "  2. Activate the environment: source venv/bin/activate"
echo "  3. Run cloaking: python src/cloak.py images/input/test_face.jpg"
echo ""
echo "For detailed instructions, see README.md"
echo ""
