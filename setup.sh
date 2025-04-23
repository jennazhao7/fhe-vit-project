#!/bin/bash
set -e

echo "🧠 Portable Setup for FHE-ViT"

# === 1. Detect Platform ===
OS="$(uname)"
echo "🔍 Detected OS: $OS"

# === 2. Install dependencies ===
echo "📦 Checking required packages..."

install_linux_dependencies() {
    sudo apt update
    sudo apt install -y build-essential cmake python3 python3-venv python3-pip
}

install_macos_dependencies() {
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh"
        exit 1
    fi
    brew install cmake python
}

if [[ "$OS" == "Linux" ]]; then
    install_linux_dependencies
elif [[ "$OS" == "Darwin" ]]; then
    install_macos_dependencies
else
    echo "❌ Unsupported OS: $OS"
    exit 1
fi

# === 3. Build OpenFHE ===
echo "🔧 [1/4] Building OpenFHE..."
cd openfhe-development
mkdir -p build && cd build
cmake ..
make -j2
sudo make install
cd ../../

# === 4. Create Python environment ===
echo "🐍 [2/4] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

if [ -f requirements.txt ]; then
    echo "📜 Installing Python dependencies..."
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found. Skipping Python dependencies."
fi

# === 5. Build C++ project ===
echo "⚙️ [3/4] Building C++ FHE pipeline..."
mkdir -p build && cd build
cmake ..
make -j2
cd ..

echo "✅ Setup complete!"
echo "➡️ To activate your environment:"
echo "   source venv/bin/activate"
echo "➡️ To run the program:"
echo "   ./build/fhe_inference   # or your binary name"
