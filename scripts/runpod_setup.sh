#!/bin/bash
# NeuroFusion-AD RunPod Pod Setup Script
# Run this once when a new pod starts
# Everything installs to /workspace/ for persistence

set -e  # Exit on error

echo "============================================"
echo "NeuroFusion-AD RunPod Setup"
echo "============================================"
echo ""

# ── 1. Verify GPU ─────────────────────────────────────────────────────────────
echo "[1/8] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# ── 2. Install Python packages ─────────────────────────────────────────────────
echo ""
echo "[2/8] Installing Python packages..."
pip install -q --upgrade pip
pip install -q \
    torch-geometric==2.5.0 \
    optuna==3.5.0 \
    shap==0.44.0 \
    lifelines==0.28.0 \
    wandb \
    anthropic \
    structlog \
    fhir.resources==7.1.0 \
    fastapi uvicorn pydantic \
    scipy scikit-learn \
    seaborn matplotlib \
    jupyter ipykernel \
    pytest pytest-cov \
    python-dotenv \
    runpodctl 2>/dev/null || pip install runpodctl

echo "✅ Packages installed"

# ── 3. Set up workspace structure ─────────────────────────────────────────────
echo ""
echo "[3/8] Setting up /workspace structure..."
mkdir -p /workspace/neurofusion-ad/{data/{raw/{adni,biohermes},processed/{adni,biohermes}},models/{checkpoints,final},notebooks/{eda,training,validation},logs}
echo "✅ Workspace structure ready"

# ── 4. Clone or pull project ───────────────────────────────────────────────────
echo ""
echo "[4/8] Setting up project code..."
if [ -d "/workspace/neurofusion-ad/.git" ]; then
    echo "  → Existing repo found, pulling latest..."
    cd /workspace/neurofusion-ad && git pull
else
    echo "  → Fresh clone needed"
    echo "  ⚠️  Run manually: git clone https://github.com/YOUR_USERNAME/neurofusion-ad.git /workspace/neurofusion-ad"
    echo "  (Skipping — repo URL not configured in this script)"
fi

# ── 5. Set up environment variables ───────────────────────────────────────────
echo ""
echo "[5/8] Setting up environment..."

# Create .env file in workspace (persists across sessions)
ENV_FILE="/workspace/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" << 'ENVEOF'
# NeuroFusion-AD Environment Variables
# Fill in your actual keys:
WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY_HERE
NEUROFUSION_DATA_DIR=/workspace/neurofusion-ad/data
NEUROFUSION_MODELS_DIR=/workspace/neurofusion-ad/models
PYTHONPATH=/workspace/neurofusion-ad
ENVEOF
    echo "  ✅ Created /workspace/.env — EDIT THIS FILE with your API keys!"
else
    echo "  ✅ /workspace/.env already exists"
fi

# Source the env file
set -a
source "$ENV_FILE"
set +a

# Add to bashrc for future sessions
if ! grep -q "source /workspace/.env" ~/.bashrc 2>/dev/null; then
    echo "source /workspace/.env" >> ~/.bashrc
    echo "export PYTHONPATH=/workspace/neurofusion-ad" >> ~/.bashrc
    echo "cd /workspace/neurofusion-ad" >> ~/.bashrc
fi

# ── 6. Configure W&B ──────────────────────────────────────────────────────────
echo ""
echo "[6/8] Configuring Weights & Biases..."
if [ -n "$WANDB_API_KEY" ] && [ "$WANDB_API_KEY" != "REPLACE_WITH_YOUR_WANDB_KEY" ]; then
    wandb login "$WANDB_API_KEY" --relogin
    echo "  ✅ W&B authenticated"
else
    echo "  ⚠️  W&B key not set in /workspace/.env — set it and run: wandb login YOUR_KEY"
fi

# ── 7. Auto-terminate safety net ──────────────────────────────────────────────
echo ""
echo "[7/8] Setting up auto-terminate (6 hours)..."
# Terminates pod after 6 hours to prevent runaway costs
# Adjust hours by changing 21600 (seconds)
nohup bash -c "sleep 21600 && echo 'Auto-terminate triggered after 6 hours' && runpodctl stop pod \$RUNPOD_POD_ID 2>/dev/null || shutdown -h now" > /workspace/auto_terminate.log 2>&1 &
echo "  ✅ Auto-terminate set for 6 hours (edit /workspace/auto_terminate.log to see status)"
echo "  To cancel: kill \$(pgrep -f 'sleep 21600')"

# ── 8. Final verification ─────────────────────────────────────────────────────
echo ""
echo "[8/8] Final verification..."
python3 << 'PYEOF'
import torch
import torch_geometric
import wandb
import shap
import lifelines
import optuna
print(f"✅ torch: {torch.__version__}")
print(f"✅ torch_geometric: {torch_geometric.__version__}")
print(f"✅ GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"✅ shap, lifelines, optuna: OK")
PYEOF

echo ""
echo "============================================"
echo "✅ RunPod setup complete!"
echo ""
echo "NEXT STEPS:"
echo "1. Edit /workspace/.env with your API keys"
echo "2. Upload your data files:"
echo "   data/raw/adni/*.csv → /workspace/neurofusion-ad/data/raw/adni/"
echo "   data/raw/biohermes/*.csv → /workspace/neurofusion-ad/data/raw/biohermes/"
echo "3. Run: cd /workspace/neurofusion-ad && claude"
echo "============================================"
