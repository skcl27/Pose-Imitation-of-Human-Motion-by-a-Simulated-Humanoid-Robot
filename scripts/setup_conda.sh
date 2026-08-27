#!/bin/bash
# Setup script: Create conda environment and install all dependencies
# Usage: bash scripts/setup_conda.sh

set -e  # Exit on first error

echo "=========================================="
echo "Setting up Conda environment (py312)"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda or Anaconda first."
    echo "See: https://docs.conda.io/projects/miniconda/en/latest/"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Repository root: $REPO_ROOT"
echo ""

# Step 1: Remove existing environment if it exists
echo "Step 1: Removing old environment (if exists)..."
conda env remove -n py312 -y 2>/dev/null || true
echo "✓ Old environment cleaned"
echo ""

# Step 2: Create environment from environment.yml
echo "Step 2: Creating conda environment from environment.yml..."
cd "$REPO_ROOT"
conda env create -f environment.yml -y
echo "✓ Conda environment created"
echo ""

# Step 3: Install TensorFlow + TensorFlow-Hub via conda's pip
# Install a GPU-enabled TensorFlow build matching this machine's CUDA/cuDNN
# driver -- see docs/RUN_INSTRUCTIONS.md step 2.4 if this pin doesn't fit.
echo "Step 3: Installing TensorFlow + TensorFlow-Hub via conda (for MeTRAbs)..."
conda run -n py312 pip install "tensorflow>=2.12,<2.16" "tensorflow-hub>=0.15,<0.17" --quiet
echo "✓ TensorFlow + TensorFlow-Hub installed"
echo ""

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
TF_VERSION=$(conda run -n py312 python -c "import tensorflow as tf; print(tf.__version__)")
TF_GPUS=$(conda run -n py312 python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))")

echo "  - TensorFlow version: $TF_VERSION"
echo "  - GPUs visible: $TF_GPUS"

if [ "$TF_GPUS" == "[]" ]; then
    echo "ERROR: No GPU visible to TensorFlow. MeTRAbs needs a GPU for real-time"
    echo "inference -- fix the CUDA/cuDNN install before continuing (or set"
    echo "pose.allow_synthetic_fallback: true in configs/default.yaml to run"
    echo "without real pose tracking)."
    exit 1
fi

echo "✓ Verification passed"
echo ""

# Step 5: Summary
echo "=========================================="
echo "Setup complete! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate py312"
echo "  2. Run camera demo:      python run.py --no-webots"
echo "  3. Full system:          python run.py"
echo ""
echo "For more details, see docs/RUN_INSTRUCTIONS.md"
echo ""
