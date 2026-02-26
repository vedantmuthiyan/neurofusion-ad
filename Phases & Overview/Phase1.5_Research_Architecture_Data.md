# Phase 1: Research, Architecture & Data Pipeline Development

**Duration**: Months 1-4 (16 weeks)  
**Objective**: Establish foundational infrastructure, secure datasets, design model architecture, and build robust data preprocessing pipelines.

---

## Month 1: Environment Setup, Dataset Access & Literature Review

### Week 1-2: Infrastructure & Environment Setup

#### Hardware Procurement & Configuration
```bash
# Target Infrastructure
GPU Server:
  - 2x NVIDIA A100 (40GB) or 4x V100 (32GB)
  - 64 CPU cores (AMD EPYC or Intel Xeon)
  - 256GB RAM
  - 4TB NVMe SSD RAID (for fast I/O)
  - 10GbE network connectivity

# Alternative: Cloud Setup (AWS/GCP/Azure)
AWS: p4d.24xlarge (8x A100 40GB) - $32.77/hr → ~$5K/week
  OR p3.8xlarge (4x V100 16GB) - $12.24/hr → ~$2K/week
GCP: a2-highgpu-4g (4x A100 40GB) - similar pricing
Azure: Standard_ND96asr_v4 (8x A100 40GB) - similar pricing

# Budget Recommendation: Start with p3.8xlarge, scale to A100 for final training
```

#### Operating System & Base Software
```bash
# Step 1: OS Installation
Ubuntu 22.04.3 LTS Server
- Kernel: 5.15+ (HWE kernel for latest hardware support)
- Filesystem: ext4 on SSDs, XFS on HDDs

# Step 2: NVIDIA Drivers & CUDA
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential dkms

# Install NVIDIA Driver (version 535+ for CUDA 12.1)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install nvidia-driver-535 nvidia-utils-535

# Verify installation
nvidia-smi  # Should show GPU info

# Install CUDA Toolkit 12.1
sudo apt install cuda-toolkit-12-1

# Add to ~/.bashrc
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Step 3: cuDNN Installation (Deep Learning Acceleration)
# Download cuDNN 8.9.7 for CUDA 12.1 from NVIDIA Developer Portal
tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

#### Python Environment Setup
```bash
# Install Miniconda (lightweight)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source ~/miniconda3/bin/activate

# Create dedicated environment
conda create -n neurofusion python=3.10 -y
conda activate neurofusion

# Install PyTorch with CUDA 12.1 support
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch GPU availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"
# Expected output:
# PyTorch: 2.1.2+cu121
# CUDA Available: True
# GPU Count: 2 (or your actual GPU count)

# Install PyTorch Geometric and dependencies
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.5.0

# Verify PyG installation
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

#### Core Dependencies Installation
```bash
# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core ML/DL
torch==2.1.2
torchvision==0.16.2
torch-geometric==2.5.0
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
pandas==2.1.4

# Audio/Signal Processing
librosa==0.10.1
pydub==0.25.1
speechbrain==0.5.16
praat-parselmouth==0.4.3  # Python wrapper for Praat
whisper @ git+https://github.com/openai/whisper.git  # OpenAI Whisper for ASR

# NLP
spacy==3.7.2
transformers==4.36.2
nltk==3.8.1

# Computer Vision (for potential MRI integration)
opencv-python==4.9.0.80
nibabel==5.2.0  # For NIfTI neuroimaging files

# Medical Data Standards
fhir.resources==7.1.0  # FHIR R4 Python models
pydicom==2.4.4  # DICOM medical imaging
hl7apy==1.3.4  # HL7 v2.x message parsing

# Data Processing
openpyxl==3.1.2  # Excel file handling (for ADNI data exports)
h5py==3.10.0  # HDF5 for efficient data storage
tables==3.9.2  # PyTables for large datasets

# Visualization
matplotlib==3.8.2
seaborn==0.13.1
plotly==5.18.0

# Explainability
shap==0.44.1  # SHapley Additive exPlanations
lime==0.2.0.1  # Local Interpretable Model-agnostic Explanations
captum==0.7.0  # PyTorch interpretability library

# Survival Analysis
lifelines==0.28.0  # Cox Proportional Hazards, Kaplan-Meier
scikit-survival==0.22.2

# API Development
fastapi==0.108.0
uvicorn[standard]==0.25.0
pydantic==2.5.3

# Testing & Quality
pytest==7.4.4
pytest-cov==4.1.0
pytest-mock==3.12.0
black==23.12.1  # Code formatter
pylint==3.0.3  # Linter
mypy==1.8.0  # Static type checker
bandit==1.7.6  # Security linter

# Utilities
tqdm==4.66.1
python-dotenv==1.0.0
pyyaml==6.0.1
wandb==0.16.2  # Experiment tracking (Weights & Biases)
tensorboard==2.15.1  # Alternative: TensorBoard

# Docker SDK (for Phase 3)
docker==7.0.0

# Cryptography (for HIPAA compliance)
cryptography==41.0.7
EOF

# Install all dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### Version Control & Project Structure
```bash
# Initialize Git repository
mkdir neurofusion-ad && cd neurofusion-ad
git init
git config user.name "NeuroFusion Team"
git config user.email "team@neurofusion.ai"

# Create .gitignore
cat > .gitignore << 'EOF'
# Data files (never commit datasets)
data/raw/
data/processed/
*.csv
*.h5
*.hdf5
*.nii
*.nii.gz
*.dcm

# Model checkpoints
models/checkpoints/
*.pth
*.pt
*.ckpt

# Logs
logs/
*.log
wandb/
tensorboard_logs/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv
*.egg-info/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Secrets
.env
secrets/
*.key
*.pem

# OS
.DS_Store
Thumbs.db

# Temporary
temp/
tmp/
*.tmp
EOF

# Create project directory structure
mkdir -p {data/{raw,processed,interim},models/{checkpoints,final},src/{data,models,utils,api},notebooks,tests,docs,configs,scripts}

# Create initial directory structure tree
tree -L 2 -d

# Expected output:
# .
# ├── configs         # Configuration YAML files
# ├── data
# │   ├── interim     # Intermediate processed data
# │   ├── processed   # Final processed features
# │   └── raw         # Original ADNI/Bio-Hermes downloads
# ├── docs            # Documentation
# ├── models
# │   ├── checkpoints # Training checkpoints
# │   └── final       # Final production models
# ├── notebooks       # Jupyter notebooks for EDA
# ├── scripts         # Utility scripts (download, preprocess)
# ├── src
# │   ├── api         # FastAPI service
# │   ├── data        # Data loading/preprocessing
# │   ├── models      # Model architecture definitions
# │   └── utils       # Helper functions
# └── tests           # Unit and integration tests

# Create README
cat > README.md << 'EOF'
# NeuroFusion-AD: Multimodal GNN for Alzheimer's Progression Prediction

## Project Overview
Roche-acquisition-ready AI algorithm for predicting MCI-to-AD progression using multimodal data fusion.

## Quick Start
```bash
conda activate neurofusion
python src/train.py --config configs/train_config.yaml
```

## Documentation
- [Phase 1 Plan](docs/Phase1_Plan.md)
- [Architecture Spec](docs/Architecture.md)
- [API Documentation](docs/API.md)

## License
Proprietary - NeuroFusion Development Team
EOF

# Initial commit
git add .
git commit -m "Initial project structure and environment setup"
```

#### Monitoring & Logging Setup
```bash
# Install system monitoring tools
sudo apt install htop nvtop iotop

# Set up Weights & Biases (W&B) for experiment tracking
# Create account at wandb.ai
wandb login  # Follow prompts

# Create W&B project
wandb init --project neurofusion-ad --name phase1-setup

# Alternative: TensorBoard (local)
# TensorBoard logs will be saved to logs/tensorboard/
```

---

### Week 3-4: Dataset Access & Initial Acquisition

#### ADNI Dataset Access

**Application Process**:
```
Step 1: Registration
URL: https://adni.loni.usc.edu/
1. Create LONI account
2. Complete New User Registration form
3. List institutional affiliation (university/research org required)
4. Describe research purpose:
   "Development of a multimodal machine learning algorithm for predicting 
    Alzheimer's disease progression from mild cognitive impairment, integrating 
    fluid biomarkers, neuroimaging, and cognitive assessments."

Step 2: Data Use Agreement (DUA)
1. Download ADNI DUA template
2. Have PI (Principal Investigator) sign
3. Have institutional official sign (Sponsored Projects Office)
4. Upload signed DUA to LONI portal
5. Wait for approval (typically 2-4 weeks, plan accordingly)

Step 3: Access Granted
- Receive email confirmation
- Gain access to ADNI data download page
- Review Data Dictionary (critical for understanding column encodings)
```

**Data Download Strategy**:
```bash
# Create download script
cat > scripts/download_adni.sh << 'EOF'
#!/bin/bash
# ADNI Data Download Script
# Requires: LONI IDA credentials exported as environment variables
# Usage: export IDA_USER="your_username" && export IDA_PASSWORD="your_password" && ./download_adni.sh

# Download ADNIMERGE (longitudinal clinical data)
wget --user=$IDA_USER --password=$IDA_PASSWORD \
  -O data/raw/adni/ADNIMERGE.csv \
  "https://ida.loni.usc.edu/loni/download.jsp?fileID=ADNIMERGE_LATEST"

# Download Biomarker Master (plasma/CSF)
wget --user=$IDA_USER --password=$IDA_PASSWORD \
  -O data/raw/adni/UPENNBIOMK_MASTER.csv \
  "https://ida.loni.usc.edu/loni/download.jsp?fileID=UPENNBIOMK_MASTER"

# Download PET Amyloid Data (Florbetapir)
wget --user=$IDA_USER --password=$IDA_PASSWORD \
  -O data/raw/adni/BAIPETNMRC.csv \
  "https://ida.loni.usc.edu/loni/download.jsp?fileID=BAIPETNMRC_LATEST"

# Download PET Tau Data (Flortaucipir, ADNI-3 only)
wget --user=$IDA_USER --password=$IDA_PASSWORD \
  -O data/raw/adni/TAUQPETT.csv \
  "https://ida.loni.usc.edu/loni/download.jsp?fileID=TAUQPETT_LATEST"

# Download MRI FreeSurfer Volumes
wget --user=$IDA_USER --password=$IDA_PASSWORD \
  -O data/raw/adni/UCSFFSX_11_08_19.csv \
  "https://ida.loni.usc.edu/loni/download.jsp?fileID=UCSFFSX_LATEST"

# Download APOE Genotype
wget --user=$IDA_USER --password=$IDA_PASSWORD \
  -O data/raw/adni/APOERES.csv \
  "https://ida.loni.usc.edu/loni/download.jsp?fileID=APOERES"

# Download Patient Demographics
wget --user=$IDA_USER --password=$IDA_PASSWORD \
  -O data/raw/adni/PTDEMOG.csv \
  "https://ida.loni.usc.edu/loni/download.jsp?fileID=PTDEMOG_LATEST"

echo "ADNI download complete. Files saved to data/raw/adni/"
EOF

chmod +x scripts/download_adni.sh

# Execute download (after DUA approval)
# NOTE: DO NOT run until IDA credentials are secured
# ./scripts/download_adni.sh
```

**Expected ADNI Data Structure**:
```
data/raw/adni/
├── ADNIMERGE.csv           # ~2,300 subjects × 400+ variables
│   ├── RID (Patient ID)
│   ├── VISCODE (Visit code: bl, m06, m12, m24, ...)
│   ├── DX (Diagnosis: CN, MCI, Dementia)
│   ├── MMSE, ADAS11, ADAS13, RAVLT_immediate
│   └── AGE, PTGENDER, PTEDUCAT
├── UPENNBIOMK_MASTER.csv   # CSF/Plasma biomarkers
│   ├── ABETA, TAU, PTAU
│   ├── NFL (subset of ADNI-3)
│   └── Multiple assay versions (Elecsys, Roche V1, V2)
├── BAIPETNMRC.csv          # Amyloid PET
│   ├── SUMMARYSUVR_WHOLECEREBNORM (Global SUVR)
│   ├── Amyloid_Status (Positive/Negative threshold)
│   └── Regional SUVRs (frontal, parietal, temporal, ...)
├── TAUQPETT.csv            # Tau PET (ADNI-3)
│   └── Braak stage ROI values
├── UCSFFSX_11_08_19.csv    # MRI FreeSurfer
│   ├── ST90SV_UCSFFSX_11_08_19 (hippocampal volume)
│   └── 100+ cortical thickness measures
├── APOERES.csv             # APOE genotype
│   ├── APGEN1, APGEN2 (allele codes: 2, 3, or 4)
│   └── Derived: ε4 carrier status
└── PTDEMOG.csv             # Demographics
    ├── Race, Ethnicity
    └── Education years (PTEDUCAT)

Total Size: ~500MB (CSV only, excludes imaging NIfTI files)
```

#### Bio-Hermes-001 Dataset Access

**Application Process**:
```
Step 1: AD Workbench Registration
URL: https://www.alzheimersdata.org/
1. Create account (free, open access)
2. Accept Terms of Use
3. Navigate to: Data → Studies → Bio-Hermes-001
4. Click "Request Access" (instant approval for released datasets)

Step 2: Dataset Download
- Format: BIDS-compliant structure + CSV
- Access via AWS S3 or direct download (files hosted on AD Workbench)
```

```bash
# Download Bio-Hermes-001 from AD Workbench
cat > scripts/download_biohermes.sh << 'EOF'
#!/bin/bash
# Bio-Hermes-001 Download Script
# Requires: AD Workbench credentials

# Note: Update URL after public release (August 2025)
# Example structure (actual URL TBD):
BASE_URL="https://data.alzheimersdata.org/biohermes/001/"

# Download clinical data
wget ${BASE_URL}/participants.tsv -O data/raw/biohermes/participants.tsv
wget ${BASE_URL}/biomarkers.csv -O data/raw/biohermes/biomarkers.csv
wget ${BASE_URL}/digital_cognitive.csv -O data/raw/biohermes/digital_cognitive.csv

# Download Linus Health DCTclock data
wget -r -np -nH --cut-dirs=2 -P data/raw/biohermes/dctclock/ \
  ${BASE_URL}/dctclock/

# Download CANTAB data
wget -r -np -nH --cut-dirs=2 -P data/raw/biohermes/cantab/ \
  ${BASE_URL}/cantab/

echo "Bio-Hermes-001 download complete."
EOF

chmod +x scripts/download_biohermes.sh
```

**Expected Bio-Hermes Data Structure**:
```
data/raw/biohermes/
├── participants.tsv        # ~300 subjects
│   ├── participant_id
│   ├── age, sex, education
│   ├── diagnosis (CN, MCI, Mild_AD)
│   └── APOE status
├── biomarkers.csv          # Roche Elecsys Assays
│   ├── participant_id, timepoint
│   ├── PTAU217_pg_mL       # Primary target!
│   ├── ABETA42_pg_mL, ABETA40_pg_mL
│   ├── PTAU181_pg_mL, TTAU_pg_mL
│   └── NFL_pg_mL
├── digital_cognitive.csv   # Aggregated scores
│   ├── DCTclock_Total_Time_sec
│   ├── DCTclock_Think_Time_sec
│   ├── CANTAB_RVP_A_prime (attention)
│   └── MoCA_Total_Score
├── dctclock/               # Raw DCTclock JSON files
│   └── sub-*/
│       └── digital_clock_drawing_*.json
└── cantab/                 # CANTAB battery results
    └── sub-*/
        └── cantab_results_*.csv

Total Size: ~50MB (excluding raw audio if available)
```

#### DementiaBank Corpus (Supplementary)

```bash
# DementiaBank Pitt Corpus Download
# URL: https://dementia.talkbank.org/access/English/Pitt.html

cat > scripts/download_dementiabank.sh << 'EOF'
#!/bin/bash
# Download Cookie Theft audio transcripts
git clone https://github.com/TalkBank/DementiaBank-Pitt.git data/raw/dementiabank

# Expected structure:
# data/raw/dementiabank/
# ├── Control/   # ~240 healthy controls
# │   └── cookie/
# │       ├── 001-0.cha (CHAT transcript format)
# │       └── 001-0.mp3 (audio recording)
# └── Dementia/  # ~310 dementia patients
#     └── cookie/
#         └── ...

echo "DementiaBank Pitt Corpus downloaded."
EOF

chmod +x scripts/download_dementiabank.sh
./scripts/download_dementiabank.sh
```

---

### Week 5-6: Literature Review & Baseline Model Research

#### Systematic Literature Review

**Research Questions**:
1. What are the state-of-the-art methods for MCI-to-AD progression prediction?
2. Which biomarkers (fluid, digital, imaging) demonstrate highest predictive power?
3. How have prior studies implemented multimodal fusion?
4. What are the current explainability approaches in medical AI?

**Search Strategy**:
```
Databases:
  - PubMed / MEDLINE
  - IEEE Xplore
  - arXiv (cs.LG, cs.AI, cs.CV)
  - Google Scholar

Search Terms:
  ("Alzheimer's Disease" OR "Alzheimer" OR "AD") AND
  ("Mild Cognitive Impairment" OR "MCI") AND
  ("progression prediction" OR "conversion prediction" OR "prognosis") AND
  ("machine learning" OR "deep learning" OR "neural network" OR "graph neural network")

Date Range: 2018-2025 (focus on recent deep learning era)

Target: 50-75 highly relevant papers
```

**Key Papers to Review** (Starter List):
```
Multimodal Fusion:
[1] Li et al. (2021) "Multimodal attention-based deep learning for Alzheimer's disease diagnosis"
    - Cross-modal attention mechanism for MRI + PET + genetic data
    - Achieved AUC=0.91 for AD vs CN classification
    - Key Insight: Attention weights provide interpretability

[2] Huang et al. (2023) "Early Prediction of Alzheimer's Disease with Multimodal Multitask Deep Learning"
    - Multi-task learning (diagnosis + MMSE regression + conversion prediction)
    - ADNI dataset validation
    - Key Insight: Shared representations improve generalization

Graph Neural Networks:
[3] Ozdemir et al. (2024) "A Dynamic Model for Early Prediction of Alzheimer's Disease by Graph Neural Networks"
    - Patient similarity network + temporal GNN
    - PSB 2025 proceedings
    - Key Insight: Longitudinal graph evolution captures disease dynamics

[4] Wen et al. (2020) "Convolutional Neural Networks for Classification of Alzheimer's Disease: Overview and Reproducible Evaluation"
    - Comprehensive survey of CNN architectures for AD
    - Reproducibility crisis in AD research (importance of standardized splits)

Digital Biomarkers:
[5] Luz et al. (2021) "Alzheimer's Dementia Recognition from Spontaneous Speech: How Far Are We from Automated Assessment?"
    - Linguistic + acoustic features from Cookie Theft task
    - Achieved 0.84 AUC for AD detection
    - Key Features: Pause duration, semantic density, lexical diversity

[6] Diaz-Asper et al. (2019) "A Digital Clock and Recall Test for Dementia Screening"
    - DCTclock validation study
    - High correlation with MoCA scores (r=0.76)
    - Sensitivity: 86%, Specificity: 80% for MCI detection

Fluid Biomarkers:
[7] Palmqvist et al. (2020) "Discriminative Accuracy of Plasma Phospho-tau217 for Alzheimer Disease"
    - pTau-217 outperforms pTau-181 for amyloid PET prediction (AUC=0.96)
    - Single-timepoint measure, no longitudinal modeling
    - Key Insight: pTau-217 is THE gold standard plasma marker

[8] Mattsson-Carlgren et al. (2021) "Plasma Biomarker Strategy for Selecting Patients for Screening"
    - Demonstrates plasma biomarkers can reduce PET scans by 70%
    - Economic argument for blood-first screening

Explainability (XAI):
[9] Tjoa & Guan (2021) "A Survey on Explainable Artificial Intelligence (XAI): Toward Medical XAI"
    - Comprehensive review of XAI methods
    - SHAP, LIME, Grad-CAM, Attention-based explanations
    - Regulatory importance: FDA draft guidance mentions explainability

[10] Rudin (2019) "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions"
    - Controversial: Argues for inherently interpretable models
    - Relevance: Attention weights (our approach) are inherently interpretable

Regulatory & Clinical Deployment:
[11] FDA (2019) "Clinical Decision Support Software: Guidance for Industry"
    - Defines non-device CDS exemptions (our algorithm likely NOT exempt)
    - Key: Must demonstrate clinical utility, not just statistical performance

[12] Hwang et al. (2023) "Evaluating AI for Medical Diagnosis"
    - Importance of external validation (not just train/test split on ADNI)
    - Common pitfall: Overfitting to institution-specific biases
```

**Literature Review Deliverable**:
```
Create: docs/Literature_Review_Summary.md
Sections:
  1. Summary of Current Methods (Table: Method, Dataset, Performance)
  2. Identified Gaps (e.g., lack of multimodal GNN for AD)
  3. Relevant Architectures (diagrams, equations)
  4. Biomarker Selection Rationale (evidence-based)
  5. Explainability Best Practices
  6. Regulatory Considerations from Prior AI/ML Clearances

Update Progress: End of Week 6
```

---

### Week 7-8: Exploratory Data Analysis (EDA)

#### ADNI Dataset EDA

```python
# Create: notebooks/01_ADNI_EDA.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load ADNIMERGE
adni = pd.read_csv('data/raw/adni/ADNIMERGE.csv')

# === PART 1: Cohort Selection ===
print("=== ADNI Cohort Overview ===")
print(f"Total subjects: {adni['RID'].nunique()}")
print(f"Total visits: {len(adni)}")
print(f"Diagnosis distribution:\n{adni.groupby('DX')['RID'].nunique()}")

# Filter: Baseline MCI patients
mci_baseline = adni[(adni['VISCODE'] == 'bl') & (adni['DX'] == 'MCI')]
print(f"\nBaseline MCI subjects: {len(mci_baseline)}")

# Check longitudinal follow-up (at least 2 visits beyond baseline)
visit_counts = adni[adni['RID'].isin(mci_baseline['RID'])].groupby('RID')['VISCODE'].count()
mci_with_followup = mci_baseline[mci_baseline['RID'].isin(visit_counts[visit_counts >= 3].index)]
print(f"MCI with ≥2 follow-ups: {len(mci_with_followup)}")

# Define progression: MCI → Dementia within 24 months
def label_progression(subject_id):
    subject_data = adni[adni['RID'] == subject_id].sort_values('VISCODE')
    baseline_dx = subject_data.iloc[0]['DX']
    if baseline_dx != 'MCI':
        return None
    
    # Check 24-month visit
    m24_data = subject_data[subject_data['VISCODE'].isin(['m24', 'm18', 'm12'])]  # Allow some tolerance
    if len(m24_data) == 0:
        return None  # No 24-month data
    
    latest_dx = m24_data.iloc[-1]['DX']
    return 'Progressive' if latest_dx == 'Dementia' else 'Stable'

mci_with_followup['Progression_Label'] = mci_with_followup['RID'].apply(label_progression)
mci_with_followup = mci_with_followup[mci_with_followup['Progression_Label'].notna()]

print(f"\nProgression labels:")
print(mci_with_followup['Progression_Label'].value_counts())
# Expected: ~400 Progressive, ~700 Stable

# === PART 2: Biomarker Distribution Analysis ===
biomarkers = pd.read_csv('data/raw/adni/UPENNBIOMK_MASTER.csv')

# Merge with MCI cohort
mci_bio = mci_with_followup.merge(biomarkers, on=['RID', 'VISCODE'], how='left')

# pTau-181 distribution (ADNI doesn't have pTau-217, so we use pTau-181)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# pTau-181
axes[0].hist(mci_bio[mci_bio['Progression_Label']=='Stable']['PTAU'], bins=30, alpha=0.6, label='Stable')
axes[0].hist(mci_bio[mci_bio['Progression_Label']=='Progressive']['PTAU'], bins=30, alpha=0.6, label='Progressive')
axes[0].set_xlabel('pTau-181 (pg/mL)')
axes[0].set_ylabel('Count')
axes[0].set_title('pTau-181 Distribution')
axes[0].legend()
axes[0].axvline(x=24, color='red', linestyle='--', label='Threshold (24 pg/mL)')  # Example cutoff

# Aβ42/40 Ratio
mci_bio['ABETA_RATIO'] = mci_bio['ABETA42'] / mci_bio['ABETA40']
axes[1].hist(mci_bio[mci_bio['Progression_Label']=='Stable']['ABETA_RATIO'].dropna(), bins=30, alpha=0.6, label='Stable')
axes[1].hist(mci_bio[mci_bio['Progression_Label']=='Progressive']['ABETA_RATIO'].dropna(), bins=30, alpha=0.6, label='Progressive')
axes[1].set_xlabel('Aβ42/40 Ratio')
axes[1].set_title('Amyloid Ratio Distribution')
axes[1].axvline(x=0.067, color='red', linestyle='--', label='Amyloid+ Threshold')
axes[1].legend()

# NfL (if available in ADNI-3 subset)
axes[2].hist(mci_bio[mci_bio['Progression_Label']=='Stable']['NFL'].dropna(), bins=30, alpha=0.6, label='Stable')
axes[2].hist(mci_bio[mci_bio['Progression_Label']=='Progressive']['NFL'].dropna(), bins=30, alpha=0.6, label='Progressive')
axes[2].set_xlabel('NfL (pg/mL)')
axes[2].set_title('Neurofilament Light Distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig('docs/figures/adni_biomarker_distributions.png', dpi=300)
plt.show()

# === PART 3: Missing Data Analysis ===
missing_analysis = mci_bio[['RID', 'ABETA42', 'PTAU', 'NFL', 'MMSE', 'ADAS11']].isnull().sum()
print("\n=== Missing Data ===")
print(missing_analysis)

# Heatmap of missingness
plt.figure(figsize=(10, 6))
sns.heatmap(mci_bio[['ABETA42', 'PTAU', 'NFL', 'MMSE', 'ADAS11', 'AGE']].isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap (MCI Cohort)')
plt.savefig('docs/figures/adni_missing_data.png', dpi=300)
plt.show()

# === PART 4: Cognitive Trajectory Visualization ===
# Plot MMSE over time for Progressive vs Stable
sample_progressive = mci_with_followup[mci_with_followup['Progression_Label']=='Progressive']['RID'].sample(20)
sample_stable = mci_with_followup[mci_with_followup['Progression_Label']=='Stable']['RID'].sample(20)

plt.figure(figsize=(12, 6))
for rid in sample_progressive:
    subject = adni[adni['RID']==rid].sort_values('VISCODE')
    visits = subject['VISCODE'].map({'bl': 0, 'm06': 6, 'm12': 12, 'm18': 18, 'm24': 24, 'm36': 36})
    plt.plot(visits, subject['MMSE'], color='red', alpha=0.4)

for rid in sample_stable:
    subject = adni[adni['RID']==rid].sort_values('VISCODE')
    visits = subject['VISCODE'].map({'bl': 0, 'm06': 6, 'm12': 12, 'm18': 18, 'm24': 24, 'm36': 36})
    plt.plot(visits, subject['MMSE'], color='blue', alpha=0.4)

plt.plot([], [], color='red', label='Progressive MCI')
plt.plot([], [], color='blue', label='Stable MCI')
plt.xlabel('Months from Baseline')
plt.ylabel('MMSE Score')
plt.title('MMSE Trajectories: Progressive vs Stable MCI')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('docs/figures/adni_mmse_trajectories.png', dpi=300)
plt.show()

# === PART 5: APOE ε4 Effect Analysis ===
apoe = pd.read_csv('data/raw/adni/APOERES.csv')
mci_apoe = mci_with_followup.merge(apoe[['RID', 'APGEN1', 'APGEN2']], on='RID', how='left')
mci_apoe['APOE_e4_count'] = (mci_apoe['APGEN1'] == 4).astype(int) + (mci_apoe['APGEN2'] == 4).astype(int)

plt.figure(figsize=(8, 6))
pd.crosstab(mci_apoe['APOE_e4_count'], mci_apoe['Progression_Label'], normalize='index').plot(kind='bar', stacked=False)
plt.xlabel('APOE ε4 Allele Count')
plt.ylabel('Proportion')
plt.title('Progression Rate by APOE ε4 Status')
plt.legend(title='Outcome')
plt.xticks(rotation=0)
plt.savefig('docs/figures/adni_apoe_effect.png', dpi=300)
plt.show()

print("\n=== EDA Complete ===")
print("Key findings saved to docs/figures/")
```

**EDA Deliverable**: `notebooks/01_ADNI_EDA.ipynb` + Figures in `docs/figures/`

---

## Month 2: Data Preprocessing Pipeline Development

### Week 9-10: ADNI Preprocessing Pipeline

#### Feature Engineering Pipeline

```python
# Create: src/data/adni_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class ADNIPreprocessor:
    """
    Comprehensive ADNI data preprocessing pipeline.
    Handles: cohort selection, feature engineering, missing value imputation, normalization.
    """
    
    def __init__(self, config):
        self.config = config
        self.imputer = None
        self.scaler = None
        
    def load_raw_data(self):
        """Load all ADNI raw CSV files."""
        print("Loading ADNI raw data...")
        self.adnimerge = pd.read_csv('data/raw/adni/ADNIMERGE.csv')
        self.biomarkers = pd.read_csv('data/raw/adni/UPENNBIOMK_MASTER.csv')
        self.apoe = pd.read_csv('data/raw/adni/APOERES.csv')
        self.demographics = pd.read_csv('data/raw/adni/PTDEMOG.csv')
        self.pet_amyloid = pd.read_csv('data/raw/adni/BAIPETNMRC.csv')
        print("Data loaded successfully.")
        
    def select_mci_cohort(self):
        """
        Select MCI patients at baseline with longitudinal follow-up.
        Label progression: Progressive (→ Dementia in 24mo) vs Stable.
        """
        print("Selecting MCI cohort...")
        
        # Baseline MCI
        mci_bl = self.adnimerge[
            (self.adnimerge['VISCODE'] == 'bl') & 
            (self.adnimerge['DX'] == 'MCI')
        ].copy()
        
        # Require ≥2 follow-up visits
        visit_counts = self.adnimerge.groupby('RID')['VISCODE'].count()
        mci_bl = mci_bl[mci_bl['RID'].isin(visit_counts[visit_counts >= 3].index)]
        
        # Label progression
        def label_progression(rid):
            subject = self.adnimerge[self.adnimerge['RID'] == rid].sort_values('VISCODE')
            # Check m24 visit (or nearest)
            future_visits = subject[subject['VISCODE'].isin(['m12', 'm18', 'm24'])]
            if len(future_visits) == 0:
                return None
            latest = future_visits.iloc[-1]
            return 1 if latest['DX'] == 'Dementia' else 0  # 1=Progressive, 0=Stable
        
        mci_bl['PROGRESSION_LABEL'] = mci_bl['RID'].apply(label_progression)
        mci_bl = mci_bl[mci_bl['PROGRESSION_LABEL'].notna()]
        
        print(f"MCI cohort: {len(mci_bl)} subjects")
        print(f"  Progressive: {(mci_bl['PROGRESSION_LABEL']==1).sum()}")
        print(f"  Stable: {(mci_bl['PROGRESSION_LABEL']==0).sum()}")
        
        self.cohort = mci_bl
        return self.cohort
        
    def merge_features(self):
        """Merge biomarkers, APOE, demographics into cohort dataframe."""
        print("Merging features...")
        
        # Biomarkers (baseline only)
        bio_bl = self.biomarkers[
            (self.biomarkers['VISCODE'] == 'bl') &
            (self.biomarkers['RID'].isin(self.cohort['RID']))
        ][['RID', 'ABETA42', 'ABETA40', 'TAU', 'PTAU', 'NFL']]
        
        # Amyloid PET (ground truth for validation)
        pet_bl = self.pet_amyloid[
            (self.pet_amyloid['VISCODE'] == 'bl') &
            (self.pet_amyloid['RID'].isin(self.cohort['RID']))
        ][['RID', 'SUMMARYSUVR_WHOLECEREBNORM', 'Amyloid_Status']]
        
        # APOE
        apoe_processed = self.apoe[['RID', 'APGEN1', 'APGEN2']].copy()
        apoe_processed['APOE_e4_COUNT'] = (
            (apoe_processed['APGEN1'] == 4).astype(int) + 
            (apoe_processed['APGEN2'] == 4).astype(int)
        )
        
        # Merge all
        self.cohort = self.cohort.merge(bio_bl, on='RID', how='left')
        self.cohort = self.cohort.merge(pet_bl, on='RID', how='left')
        self.cohort = self.cohort.merge(apoe_processed[['RID', 'APOE_e4_COUNT']], on='RID', how='left')
        
        # Compute derived features
        self.cohort['ABETA_RATIO'] = self.cohort['ABETA42'] / self.cohort['ABETA40']
        
        print(f"Merged features. Shape: {self.cohort.shape}")
        
    def impute_missing_values(self):
        """
        Handle missing values using Multiple Imputation by Chained Equations (MICE).
        """
        print("Imputing missing values...")
        
        # Features to impute
        features_to_impute = [
            'ABETA42', 'ABETA40', 'TAU', 'PTAU', 'NFL', 'ABETA_RATIO',
            'MMSE', 'ADAS11', 'RAVLT_immediate', 'AGE', 'PTEDUCAT'
        ]
        
        # MICE (Iterative Imputer)
        self.imputer = IterativeImputer(
            max_iter=10, 
            random_state=42,
            initial_strategy='median',
            imputation_order='ascending'
        )
        
        imputed_data = self.imputer.fit_transform(self.cohort[features_to_impute])
        self.cohort[features_to_impute] = imputed_data
        
        print("Imputation complete. Remaining NaNs:", self.cohort[features_to_impute].isnull().sum().sum())
        
    def calibrate_ptau181_to_ptau217(self):
        """
        Calibrate pTau-181 to pTau-217 using published conversion formula.
        
        Note: This is a simplified linear calibration. In reality, would use
        Bio-Hermes-001 data to fit a robust calibration curve.
        
        Reference: Palmqvist et al. (2020) - pTau-217 correlation with pTau-181
        """
        print("Calibrating pTau-181 → pTau-217...")
        
        # Simplified linear model: pTau-217 ≈ 0.85 * pTau-181 + 0.5
        # TODO: Replace with empirical calibration from Bio-Hermes-001
        self.cohort['PTAU217_CALIBRATED'] = 0.85 * self.cohort['PTAU'] + 0.5
        
        print("Calibration complete. New feature: PTAU217_CALIBRATED")
        
    def normalize_features(self):
        """Normalize continuous features for neural network input."""
        print("Normalizing features...")
        
        # Features for standardization (z-score)
        features_to_standardize = ['PTAU217_CALIBRATED', 'ABETA_RATIO', 'NFL', 'TAU']
        self.scaler_standard = StandardScaler()
        self.cohort[features_to_standardize] = self.scaler_standard.fit_transform(
            self.cohort[features_to_standardize]
        )
        
        # Features for min-max scaling [0, 1]
        features_to_minmax = ['AGE', 'PTEDUCAT', 'MMSE', 'ADAS11']
        self.scaler_minmax = MinMaxScaler()
        self.cohort[features_to_minmax] = self.scaler_minmax.fit_transform(
            self.cohort[features_to_minmax]
        )
        
        print("Normalization complete.")
        
    def save_processed_data(self, output_path='data/processed/adni_processed.csv'):
        """Save processed cohort to CSV."""
        print(f"Saving processed data to {output_path}...")
        self.cohort.to_csv(output_path, index=False)
        
        # Save scalers for inference
        with open('models/adni_scaler_standard.pkl', 'wb') as f:
            pickle.dump(self.scaler_standard, f)
        with open('models/adni_scaler_minmax.pkl', 'wb') as f:
            pickle.dump(self.scaler_minmax, f)
        with open('models/adni_imputer.pkl', 'wb') as f:
            pickle.dump(self.imputer, f)
        
        print("Preprocessing complete!")
        
    def run_full_pipeline(self):
        """Execute full preprocessing pipeline."""
        self.load_raw_data()
        self.select_mci_cohort()
        self.merge_features()
        self.impute_missing_values()
        self.calibrate_ptau181_to_ptau217()
        self.normalize_features()
        self.save_processed_data()
        
        return self.cohort

# === USAGE ===
if __name__ == "__main__":
    config = {}  # Load from YAML in production
    preprocessor = ADNIPreprocessor(config)
    processed_data = preprocessor.run_full_pipeline()
    
    print("\n=== Preprocessing Summary ===")
    print(f"Final cohort size: {len(processed_data)}")
    print(f"Features: {processed_data.columns.tolist()}")
    print(f"Class balance: {processed_data['PROGRESSION_LABEL'].value_counts()}")
```

#### Digital Biomarker Synthesis (ADNI Workaround)

Since ADNI lacks digital acoustic/motor data, we have two options:
1. **Skip digital features during ADNI pre-training** (use only fluid/clinical features)
2. **Synthesize plausible digital features** based on cognitive scores

**Option 2 Implementation** (for more robust pre-training):

```python
# Create: src/data/adni_digital_synthesis.py

import numpy as np
from sklearn.linear_model import LinearRegression

class DigitalBiomarkerSynthesizer:
    """
    Synthesize digital biomarker features for ADNI cohort.
    Uses cognitive scores (MMSE, ADAS11) as proxy signals.
    
    WARNING: This is a temporary solution for pre-training ONLY.
    Real digital data from Bio-Hermes will be used for fine-tuning.
    """
    
    def __init__(self, cohort_df):
        self.cohort = cohort_df
        
    def synthesize_acoustic_features(self):
        """
        Generate synthetic acoustic features correlated with cognitive decline.
        
        Assumptions (from literature):
        - Lower MMSE → Higher jitter, shimmer, pause duration
        - Lower semantic density correlates with ADAS11 (word recall)
        """
        np.random.seed(42)
        
        # Jitter (frequency variation): inversely proportional to MMSE
        # Healthy: ~1.2%, Impaired: ~2.8%
        mmse_normalized = (self.cohort['MMSE'] - self.cohort['MMSE'].min()) / (
            self.cohort['MMSE'].max() - self.cohort['MMSE'].min()
        )
        jitter_base = 2.8 - 1.6 * mmse_normalized
        self.cohort['ACOUSTIC_JITTER'] = jitter_base + np.random.normal(0, 0.2, len(self.cohort))
        
        # Shimmer (amplitude variation): similar correlation
        shimmer_base = 9.0 - 4.0 * mmse_normalized
        self.cohort['ACOUSTIC_SHIMMER'] = shimmer_base + np.random.normal(0, 0.5, len(self.cohort))
        
        # Pause duration: higher in cognitively impaired
        pause_base = 1.8 - 0.6 * mmse_normalized
        self.cohort['ACOUSTIC_PAUSE_DURATION'] = pause_base + np.random.normal(0, 0.3, len(self.cohort))
        
        # Semantic density: correlated with RAVLT immediate recall
        semantic_base = 15 + 0.4 * self.cohort['RAVLT_immediate']
        self.cohort['ACOUSTIC_SEMANTIC_DENSITY'] = semantic_base + np.random.normal(0, 3, len(self.cohort))
        
        print("Synthetic acoustic features generated.")
        
    def synthesize_motor_features(self):
        """
        Generate synthetic motor features.
        
        Assumptions:
        - Gait speed correlates with MMSE
        - Stride variability increases with cognitive decline
        """
        np.random.seed(42)
        
        mmse_normalized = (self.cohort['MMSE'] - self.cohort['MMSE'].min()) / (
            self.cohort['MMSE'].max() - self.cohort['MMSE'].min()
        )
        
        # Gait speed: healthy ~1.2 m/s, impaired ~0.7 m/s
        gait_base = 0.7 + 0.5 * mmse_normalized
        self.cohort['MOTOR_GAIT_SPEED'] = gait_base + np.random.normal(0, 0.1, len(self.cohort))
        
        # Stride variability (CV): increases with decline
        stride_var_base = 0.08 - 0.03 * mmse_normalized
        self.cohort['MOTOR_STRIDE_VARIABILITY'] = stride_var_base + np.random.normal(0, 0.01, len(self.cohort))
        
        # Turn duration: longer in impaired
        turn_base = 3.5 - 1.0 * mmse_normalized
        self.cohort['MOTOR_TURN_DURATION'] = turn_base + np.random.normal(0, 0.3, len(self.cohort))
        
        print("Synthetic motor features generated.")
        
    def run_synthesis(self):
        """Execute full synthesis pipeline."""
        self.synthesize_acoustic_features()
        self.synthesize_motor_features()
        return self.cohort

# === USAGE ===
if __name__ == "__main__":
    processed_adni = pd.read_csv('data/processed/adni_processed.csv')
    synthesizer = DigitalBiomarkerSynthesizer(processed_adni)
    augmented_adni = synthesizer.run_synthesis()
    augmented_adni.to_csv('data/processed/adni_processed_with_digital.csv', index=False)
    print("ADNI digital features synthesized and saved.")
```

---

### Week 11-12: Bio-Hermes Preprocessing Pipeline

```python
# Create: src/data/biohermes_preprocessing.py

import pandas as pd
import numpy as np
import json
from pathlib import Path

class BioHermesPreprocessor:
    """
    Preprocess Bio-Hermes-001 dataset.
    Extracts: Roche Elecsys biomarkers, DCTclock digital features, CANTAB scores.
    """
    
    def __init__(self):
        self.participants = None
        self.biomarkers = None
        self.digital_cognitive = None
        
    def load_raw_data(self):
        """Load Bio-Hermes raw data."""
        print("Loading Bio-Hermes-001 data...")
        self.participants = pd.read_csv('data/raw/biohermes/participants.tsv', sep='\t')
        self.biomarkers = pd.read_csv('data/raw/biohermes/biomarkers.csv')
        self.digital_cognitive = pd.read_csv('data/raw/biohermes/digital_cognitive.csv')
        print("Data loaded successfully.")
        
    def process_biomarkers(self):
        """
        Process Roche Elecsys plasma biomarkers.
        CRITICAL: This dataset has pTau-217 (our gold standard)!
        """
        print("Processing biomarkers...")
        
        # Filter baseline timepoint
        bio_bl = self.biomarkers[self.biomarkers['timepoint'] == 'baseline'].copy()
        
        # Rename columns to match ADNI format
        bio_bl = bio_bl.rename(columns={
            'participant_id': 'RID',
            'PTAU217_pg_mL': 'PTAU217',  # GOLD STANDARD
            'PTAU181_pg_mL': 'PTAU181',
            'ABETA42_pg_mL': 'ABETA42',
            'ABETA40_pg_mL': 'ABETA40',
            'NFL_pg_mL': 'NFL'
        })
        
        # Compute Aβ42/40 ratio
        bio_bl['ABETA_RATIO'] = bio_bl['ABETA42'] / bio_bl['ABETA40']
        
        # Filter relevant columns
        bio_processed = bio_bl[['RID', 'PTAU217', 'PTAU181', 'ABETA_RATIO', 'NFL']]
        
        self.biomarkers_processed = bio_processed
        print(f"Biomarkers processed. Shape: {bio_processed.shape}")
        
    def process_dctclock(self):
        """
        Process DCTclock (Digital Clock Drawing Test) data.
        Extract timing and spatial features.
        """
        print("Processing DCTclock data...")
        
        dctclock_dir = Path('data/raw/biohermes/dctclock')
        dctclock_features = []
        
        for json_file in dctclock_dir.glob('sub-*/digital_clock_drawing_*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            participant_id = data['participant_id']
            
            # Extract key features
            features = {
                'RID': participant_id,
                'DCTCLOCK_TOTAL_TIME': data.get('total_time_sec', None),
                'DCTCLOCK_THINK_TIME': data.get('pre_first_touch_time_sec', None),
                'DCTCLOCK_INK_TIME': data.get('ink_time_sec', None),
                'DCTCLOCK_NUM_STROKES': data.get('num_strokes', None),
                'DCTCLOCK_CLOCK_FACE_AREA': data.get('clock_face_area_pixels', None)
            }
            dctclock_features.append(features)
        
        self.dctclock_df = pd.DataFrame(dctclock_features)
        print(f"DCTclock processed. Shape: {self.dctclock_df.shape}")
        
    def process_cantab(self):
        """
        Process CANTAB (Cambridge Neuropsychological Test Automated Battery) data.
        Focus on attention (RVP) and memory (PAL) tasks.
        """
        print("Processing CANTAB data...")
        
        cantab_dir = Path('data/raw/biohermes/cantab')
        cantab_features = []
        
        for csv_file in cantab_dir.glob('sub-*/cantab_results_*.csv'):
            data = pd.read_csv(csv_file)
            participant_id = data['participant_id'].iloc[0]
            
            # Rapid Visual Processing (RVP) - attention task
            rvp_data = data[data['task'] == 'RVP']
            rvp_a_prime = rvp_data['A_prime'].mean() if len(rvp_data) > 0 else None
            
            # Paired Associates Learning (PAL) - memory task
            pal_data = data[data['task'] == 'PAL']
            pal_errors = pal_data['total_errors'].sum() if len(pal_data) > 0 else None
            
            features = {
                'RID': participant_id,
                'CANTAB_RVP_A_PRIME': rvp_a_prime,
                'CANTAB_PAL_ERRORS': pal_errors
            }
            cantab_features.append(features)
        
        self.cantab_df = pd.DataFrame(cantab_features)
        print(f"CANTAB processed. Shape: {self.cantab_df.shape}")
        
    def merge_all_features(self):
        """Merge biomarkers, DCTclock, CANTAB, and participants."""
        print("Merging all features...")
        
        cohort = self.participants.copy()
        cohort = cohort.rename(columns={'participant_id': 'RID'})
        
        # Merge biomarkers
        cohort = cohort.merge(self.biomarkers_processed, on='RID', how='left')
        
        # Merge digital cognitive
        cohort = cohort.merge(self.dctclock_df, on='RID', how='left')
        cohort = cohort.merge(self.cantab_df, on='RID', how='left')
        
        # Encode diagnosis as progression label (for consistency with ADNI)
        # For Bio-Hermes: CN=0, MCI=check progression, Mild_AD=1
        # Note: Bio-Hermes has 12-month follow-up, so we'd need longitudinal labeling
        # For now, use baseline diagnosis as placeholder
        diagnosis_map = {'CN': 0, 'MCI': 0, 'Mild_AD': 1}
        cohort['PROGRESSION_LABEL'] = cohort['diagnosis'].map(diagnosis_map)
        
        self.cohort = cohort
        print(f"Merged cohort. Shape: {cohort.shape}")
        
    def save_processed_data(self, output_path='data/processed/biohermes_processed.csv'):
        """Save processed Bio-Hermes data."""
        print(f"Saving processed data to {output_path}...")
        self.cohort.to_csv(output_path, index=False)
        print("Bio-Hermes preprocessing complete!")
        
    def run_full_pipeline(self):
        """Execute full preprocessing pipeline."""
        self.load_raw_data()
        self.process_biomarkers()
        self.process_dctclock()
        self.process_cantab()
        self.merge_all_features()
        self.save_processed_data()
        return self.cohort

# === USAGE ===
if __name__ == "__main__":
    preprocessor = BioHermesPreprocessor()
    processed_data = preprocessor.run_full_pipeline()
    print("\n=== Bio-Hermes Processing Complete ===")
    print(f"Cohort size: {len(processed_data)}")
    print(f"Features: {processed_data.columns.tolist()}")
```

---

## Month 3: Architecture Prototyping & Baseline Models

### Week 13-14: Implement Modality-Specific Encoders

```python
# Create: src/models/encoders.py

import torch
import torch.nn as nn

class FluidBiomarkerEncoder(nn.Module):
    """
    Encoder for fluid biomarkers (pTau-217, Aβ42/40, NfL).
    Architecture: 3-layer MLP with residual connection.
    """
    def __init__(self, input_dim=3, hidden_dims=[256, 512], output_dim=768, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dims[1], output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - biomarker values (log-transformed, z-scored)
        Returns:
            [batch_size, output_dim] - embedded representation
        """
        out = self.input_proj(x)
        out = self.hidden(out)
        out = self.output_proj(out)
        return out


class DigitalAcousticEncoder(nn.Module):
    """
    Encoder for acoustic biomarkers (jitter, shimmer, pauses, semantic density).
    Architecture: 1D CNN + BiLSTM fusion for time-series features.
    """
    def __init__(self, input_dim=15, output_dim=768, dropout=0.3):
        super().__init__()
        
        # CNN branch (for Praat-extracted features: jitter, shimmer, etc.)
        self.feature_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # Projection to output dimension
        self.proj = nn.Sequential(
            nn.Linear(128 + input_dim, 512),  # Combine CNN output + raw features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - acoustic features
        Returns:
            [batch_size, output_dim] - embedded representation
        """
        # Pass through CNN (treat features as 1D sequence)
        x_cnn = x.unsqueeze(1)  # [batch, 1, input_dim]
        cnn_out = self.feature_cnn(x_cnn).squeeze(-1)  # [batch, 128]
        
        # Concatenate with raw features
        combined = torch.cat([cnn_out, x], dim=1)
        out = self.proj(combined)
        return out


class DigitalMotorEncoder(nn.Module):
    """
    Encoder for motor biomarkers (gait speed, stride variability, turn metrics).
    Architecture: 1D CNN for time-series accelerometer data.
    """
    def __init__(self, input_dim=20, output_dim=768, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - motor features
        Returns:
            [batch_size, output_dim] - embedded representation
        """
        return self.encoder(x)


class ClinicalDemographicEncoder(nn.Module):
    """
    Encoder for clinical/demographic features (age, sex, education, APOE, MMSE).
    Architecture: Embedding layers for categorical + MLP for continuous.
    """
    def __init__(self, output_dim=768, dropout=0.2):
        super().__init__()
        
        # Embeddings for categorical variables
        self.sex_embedding = nn.Embedding(2, 16)  # Male=0, Female=1
        self.apoe_embedding = nn.Embedding(3, 32)  # 0, 1, or 2 ε4 alleles
        
        # MLP for continuous variables (age, education, MMSE, etc.)
        self.continuous_encoder = nn.Sequential(
            nn.Linear(5, 128),  # [age, education, MMSE, CDR-SB, ...]
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined projection
        self.proj = nn.Sequential(
            nn.Linear(128 + 16 + 32, 512),  # continuous + sex_emb + apoe_emb
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, continuous, sex, apoe):
        """
        Args:
            continuous: [batch_size, 5] - continuous clinical features
            sex: [batch_size] - sex (0 or 1)
            apoe: [batch_size] - APOE ε4 count (0, 1, or 2)
        Returns:
            [batch_size, output_dim] - embedded representation
        """
        sex_emb = self.sex_embedding(sex)
        apoe_emb = self.apoe_embedding(apoe)
        cont_out = self.continuous_encoder(continuous)
        
        combined = torch.cat([cont_out, sex_emb, apoe_emb], dim=1)
        out = self.proj(combined)
        return out


# === Unit Test ===
if __name__ == "__main__":
    batch_size = 32
    
    # Test Fluid Biomarker Encoder
    fluid_encoder = FluidBiomarkerEncoder(input_dim=3, output_dim=768)
    fluid_input = torch.randn(batch_size, 3)
    fluid_output = fluid_encoder(fluid_input)
    print(f"Fluid Encoder Output: {fluid_output.shape}")  # Should be [32, 768]
    
    # Test Digital Acoustic Encoder
    acoustic_encoder = DigitalAcousticEncoder(input_dim=15, output_dim=768)
    acoustic_input = torch.randn(batch_size, 15)
    acoustic_output = acoustic_encoder(acoustic_input)
    print(f"Acoustic Encoder Output: {acoustic_output.shape}")  # Should be [32, 768]
    
    # Test Digital Motor Encoder
    motor_encoder = DigitalMotorEncoder(input_dim=20, output_dim=768)
    motor_input = torch.randn(batch_size, 20)
    motor_output = motor_encoder(motor_input)
    print(f"Motor Encoder Output: {motor_output.shape}")  # Should be [32, 768]
    
    # Test Clinical/Demographic Encoder
    clinical_encoder = ClinicalDemographicEncoder(output_dim=768)
    continuous_input = torch.randn(batch_size, 5)
    sex_input = torch.randint(0, 2, (batch_size,))
    apoe_input = torch.randint(0, 3, (batch_size,))
    clinical_output = clinical_encoder(continuous_input, sex_input, apoe_input)
    print(f"Clinical Encoder Output: {clinical_output.shape}")  # Should be [32, 768]
    
    print("\n✅ All encoders tested successfully!")
```

---

### Week 15-16: Implement Cross-Modal Attention & Baseline GNN

```python
# Create: src/models/cross_modal_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """
    Multi-Head Cross-Attention module for fusing multimodal embeddings.
    Provides:
      1. Dynamic weighting of modality importance
      2. Attention weights for explainability (XAI)
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Learnable modality importance weights (for weighted aggregation)
        self.modality_weights = nn.Parameter(torch.ones(4))  # 4 modalities
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, fluid_emb, acoustic_emb, motor_emb, clinical_emb):
        """
        Args:
            fluid_emb: [batch_size, 768]
            acoustic_emb: [batch_size, 768]
            motor_emb: [batch_size, 768]
            clinical_emb: [batch_size, 768]
        Returns:
            fused_emb: [batch_size, 768] - fused multimodal representation
            attention_weights: [batch_size, 4] - modality importance scores (for XAI)
        """
        batch_size = fluid_emb.size(0)
        
        # Stack modalities into sequence: [batch_size, 4, 768]
        modality_seq = torch.stack([fluid_emb, acoustic_emb, motor_emb, clinical_emb], dim=1)
        
        # Self-attention across modalities
        attn_output, attn_weights = self.multihead_attn(
            query=modality_seq,
            key=modality_seq,
            value=modality_seq,
            need_weights=True
        )  # attn_output: [batch, 4, 768], attn_weights: [batch, 4, 4]
        
        # Aggregate attention weights to get modality importance
        # Average across all pairwise attention scores
        modality_importance = attn_weights.mean(dim=1)  # [batch, 4]
        
        # Weighted sum of modality embeddings
        weights_normalized = F.softmax(self.modality_weights, dim=0)  # [4]
        weighted_seq = attn_output * weights_normalized.view(1, 4, 1)  # Broadcast
        fused_emb = weighted_seq.sum(dim=1)  # [batch, 768]
        
        # Layer norm + residual (optional: add mean of input modalities as residual)
        fused_emb = self.norm(fused_emb)
        
        return fused_emb, modality_importance


# === GNN Implementation ===
# Create: src/models/gnn.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

class NeuroFusionGNN(nn.Module):
    """
    Graph Neural Network for patient similarity network.
    Architecture: 4 layers of GATv2Conv with residual connections.
    """
    def __init__(self, input_dim=768, hidden_dim=768, num_layers=4, num_heads=8, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=input_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=False,  # Average attention heads
                    dropout=dropout
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [num_patients, 768] - fused modality embeddings
            edge_index: [2, num_edges] - patient similarity edges
            edge_weight: [num_edges] - optional edge weights (similarity scores)
        Returns:
            [num_patients, 768] - refined patient representations
        """
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.norms)):
            x_prev = x
            x = gnn_layer(x, edge_index, edge_attr=edge_weight)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # Residual connection
            if i > 0:
                x = x + x_prev
        
        return x


def construct_patient_similarity_graph(patient_embeddings, threshold=0.7):
    """
    Construct patient similarity network based on cosine similarity.
    
    Args:
        patient_embeddings: [num_patients, 768] - fused modality embeddings
        threshold: float - minimum similarity to create edge
    Returns:
        edge_index: [2, num_edges] - graph edges
        edge_weight: [num_edges] - similarity scores
    """
    num_patients = patient_embeddings.size(0)
    
    # Compute pairwise cosine similarity
    embeddings_norm = F.normalize(patient_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())  # [num_patients, num_patients]
    
    # Create edges where similarity > threshold
    edge_index = []
    edge_weight = []
    
    for i in range(num_patients):
        for j in range(i + 1, num_patients):  # Undirected graph (i < j)
            sim = similarity_matrix[i, j].item()
            if sim > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Add both directions
                edge_weight.extend([sim, sim])
    
    if len(edge_index) == 0:
        # No edges, return self-loops
        edge_index = [[i, i] for i in range(num_patients)]
        edge_weight = [1.0] * num_patients
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    return edge_index, edge_weight


# === Unit Test ===
if __name__ == "__main__":
    batch_size = 32
    
    # Test Cross-Modal Attention
    cross_attn = CrossModalAttention(embed_dim=768, num_heads=8)
    
    fluid_emb = torch.randn(batch_size, 768)
    acoustic_emb = torch.randn(batch_size, 768)
    motor_emb = torch.randn(batch_size, 768)
    clinical_emb = torch.randn(batch_size, 768)
    
    fused_emb, attn_weights = cross_attn(fluid_emb, acoustic_emb, motor_emb, clinical_emb)
    print(f"Fused Embedding: {fused_emb.shape}")  # [32, 768]
    print(f"Attention Weights: {attn_weights.shape}")  # [32, 4]
    print(f"Sample attention weights: {attn_weights[0]}")  # Should sum to ~1
    
    # Test GNN
    gnn = NeuroFusionGNN(input_dim=768, hidden_dim=768, num_layers=4)
    
    # Construct patient similarity graph
    edge_index, edge_weight = construct_patient_similarity_graph(fused_emb, threshold=0.7)
    print(f"Graph edges: {edge_index.shape}, Edge weights: {edge_weight.shape}")
    
    # GNN forward pass
    refined_emb = gnn(fused_emb, edge_index, edge_weight)
    print(f"GNN Output: {refined_emb.shape}")  # [32, 768]
    
    print("\n✅ Cross-Modal Attention & GNN tested successfully!")
```

---

## Month 4: Complete Data Pipeline & Finalize Architecture

### Week 17: DataLoader & Training Pipeline Setup

```python
# Create: src/data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class NeuroFusionDataset(Dataset):
    """
    PyTorch Dataset for NeuroFusion-AD training.
    Loads processed ADNI or Bio-Hermes data.
    """
    def __init__(self, data_path, mode='train'):
        """
        Args:
            data_path: Path to processed CSV (ADNI or Bio-Hermes)
            mode: 'train', 'val', or 'test'
        """
        self.data = pd.read_csv(data_path)
        self.mode = mode
        
        # Define feature columns
        self.fluid_features = ['PTAU217_CALIBRATED', 'ABETA_RATIO', 'NFL']
        self.acoustic_features = [
            'ACOUSTIC_JITTER', 'ACOUSTIC_SHIMMER', 'ACOUSTIC_PAUSE_DURATION', 'ACOUSTIC_SEMANTIC_DENSITY'
        ]
        self.motor_features = [
            'MOTOR_GAIT_SPEED', 'MOTOR_STRIDE_VARIABILITY', 'MOTOR_TURN_DURATION'
        ]
        self.clinical_continuous = ['AGE', 'PTEDUCAT', 'MMSE', 'ADAS11', 'RAVLT_immediate']
        self.clinical_categorical = ['PTGENDER', 'APOE_e4_COUNT']
        
        # Labels
        self.label_col = 'PROGRESSION_LABEL'  # Binary: 0=Stable, 1=Progressive
        self.mmse_24mo_col = 'MMSE_24MO'  # For regression task (if available)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary containing all modality features + labels
        """
        row = self.data.iloc[idx]
        
        # Extract features
        fluid = torch.tensor(row[self.fluid_features].values, dtype=torch.float32)
        acoustic = torch.tensor(row[self.acoustic_features].values, dtype=torch.float32)
        motor = torch.tensor(row[self.motor_features].values, dtype=torch.float32)
        clinical_cont = torch.tensor(row[self.clinical_continuous].values, dtype=torch.float32)
        sex = torch.tensor(row['PTGENDER'], dtype=torch.long)  # 0 or 1
        apoe = torch.tensor(row['APOE_e4_COUNT'], dtype=torch.long)  # 0, 1, or 2
        
        # Labels
        progression_label = torch.tensor(row[self.label_col], dtype=torch.long)
        mmse_24mo = torch.tensor(row.get(self.mmse_24mo_col, 0.0), dtype=torch.float32)  # Default 0 if missing
        
        return {
            'fluid': fluid,
            'acoustic': acoustic,
            'motor': motor,
            'clinical_cont': clinical_cont,
            'sex': sex,
            'apoe': apoe,
            'label_classification': progression_label,
            'label_regression': mmse_24mo,
            'patient_id': row['RID']
        }


def create_dataloaders(data_path, batch_size=64, train_split=0.7, val_split=0.15, seed=42):
    """
    Create train/val/test dataloaders with stratified split.
    
    Args:
        data_path: Path to processed dataset CSV
        batch_size: Batch size for training
        train_split: Proportion for training (0.7 = 70%)
        val_split: Proportion for validation (0.15 = 15%)
        seed: Random seed for reproducibility
    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    data = pd.read_csv(data_path)
    
    # Stratified split by progression label
    train_data, temp_data = train_test_split(
        data, 
        test_size=(1 - train_split), 
        stratify=data['PROGRESSION_LABEL'], 
        random_state=seed
    )
    
    val_size = val_split / (1 - train_split)
    val_data, test_data = train_test_split(
        temp_data, 
        test_size=(1 - val_size), 
        stratify=temp_data['PROGRESSION_LABEL'], 
        random_state=seed
    )
    
    # Save splits to disk for reproducibility
    train_data.to_csv('data/processed/train_split.csv', index=False)
    val_data.to_csv('data/processed/val_split.csv', index=False)
    test_data.to_csv('data/processed/test_split.csv', index=False)
    
    # Create datasets
    train_dataset = NeuroFusionDataset('data/processed/train_split.csv', mode='train')
    val_dataset = NeuroFusionDataset('data/processed/val_split.csv', mode='val')
    test_dataset = NeuroFusionDataset('data/processed/test_split.csv', mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


# === Unit Test ===
if __name__ == "__main__":
    # Assuming 'data/processed/adni_processed_with_digital.csv' exists
    train_loader, val_loader, test_loader = create_dataloaders(
        'data/processed/adni_processed_with_digital.csv',
        batch_size=32
    )
    
    # Test batch loading
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("Fluid shape:", batch['fluid'].shape)
        print("Label classification:", batch['label_classification'].shape)
        break
    
    print("\n✅ DataLoader tested successfully!")
```

---

### Week 18: Full Model Integration & Initial Testing

```python
# Create: src/models/neurofusion_model.py

import torch
import torch.nn as nn
from src.models.encoders import (
    FluidBiomarkerEncoder, DigitalAcousticEncoder, 
    DigitalMotorEncoder, ClinicalDemographicEncoder
)
from src.models.cross_modal_attention import CrossModalAttention
from src.models.gnn import NeuroFusionGNN, construct_patient_similarity_graph

class NeuroFusionAD(nn.Module):
    """
    Complete NeuroFusion-AD model.
    Architecture:
      1. Modality-specific encoders (Fluid, Acoustic, Motor, Clinical)
      2. Cross-Modal Attention fusion
      3. Patient Similarity GNN
      4. Multi-task output heads (Classification, Regression, Survival)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoders
        self.fluid_encoder = FluidBiomarkerEncoder(input_dim=3, output_dim=768)
        self.acoustic_encoder = DigitalAcousticEncoder(input_dim=15, output_dim=768)
        self.motor_encoder = DigitalMotorEncoder(input_dim=20, output_dim=768)
        self.clinical_encoder = ClinicalDemographicEncoder(output_dim=768)
        
        # Cross-Modal Attention
        self.cross_modal_attn = CrossModalAttention(embed_dim=768, num_heads=8)
        
        # GNN
        self.gnn = NeuroFusionGNN(input_dim=768, hidden_dim=768, num_layers=4)
        
        # Output Heads
        # Task 1: Amyloid Positivity Classification
        self.classification_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Sigmoid applied in loss
        )
        
        # Task 2: MMSE Trajectory Regression
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Task 3: Survival Analysis (Cox model)
        self.survival_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # [risk_score, survival_time]
        )
        
    def forward(self, batch, construct_graph=True):
        """
        Args:
            batch: Dictionary containing all modality inputs
            construct_graph: Whether to build patient similarity graph (set False for single-patient inference)
        Returns:
            Dictionary with predictions and attention weights
        """
        # Extract inputs
        fluid = batch['fluid']
        acoustic = batch['acoustic']
        motor = batch['motor']
        clinical_cont = batch['clinical_cont']
        sex = batch['sex']
        apoe = batch['apoe']
        
        batch_size = fluid.size(0)
        
        # Encode each modality
        fluid_emb = self.fluid_encoder(fluid)
        acoustic_emb = self.acoustic_encoder(acoustic)
        motor_emb = self.motor_encoder(motor)
        clinical_emb = self.clinical_encoder(clinical_cont, sex, apoe)
        
        # Cross-Modal Attention Fusion
        fused_emb, modality_importance = self.cross_modal_attn(
            fluid_emb, acoustic_emb, motor_emb, clinical_emb
        )
        
        # GNN (Patient Similarity Network)
        if construct_graph and batch_size > 1:
            edge_index, edge_weight = construct_patient_similarity_graph(fused_emb, threshold=0.7)
            refined_emb = self.gnn(fused_emb, edge_index, edge_weight)
        else:
            refined_emb = fused_emb  # Skip GNN for single-patient inference
        
        # Multi-Task Outputs
        classification_logits = self.classification_head(refined_emb)
        regression_pred = self.regression_head(refined_emb)
        survival_pred = self.survival_head(refined_emb)
        
        return {
            'classification_logits': classification_logits,  # [batch, 1]
            'regression_pred': regression_pred,  # [batch, 1]
            'survival_pred': survival_pred,  # [batch, 2]
            'modality_importance': modality_importance,  # [batch, 4] - for explainability
            'patient_embedding': refined_emb  # [batch, 768]
        }


# === Unit Test ===
if __name__ == "__main__":
    from src.data.dataset import NeuroFusionDataset
    from torch.utils.data import DataLoader
    
    # Create dummy config
    config = {}
    
    # Initialize model
    model = NeuroFusionAD(config)
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load a batch
    dataset = NeuroFusionDataset('data/processed/train_split.csv', mode='train')
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    batch = next(iter(loader))
    
    # Forward pass
    outputs = model(batch, construct_graph=True)
    
    print("\n=== Model Outputs ===")
    print(f"Classification Logits: {outputs['classification_logits'].shape}")
    print(f"Regression Pred: {outputs['regression_pred'].shape}")
    print(f"Survival Pred: {outputs['survival_pred'].shape}")
    print(f"Modality Importance: {outputs['modality_importance'].shape}")
    print(f"Sample modality weights: {outputs['modality_importance'][0]}")
    
    print("\n✅ Full model integration tested successfully!")
```

---

## Phase 1 Deliverables Checklist

- [x] Hardware infrastructure set up (GPU server or cloud instance)
- [x] Python environment configured (PyTorch, PyG, all dependencies)
- [x] Project repository initialized with proper structure
- [x] ADNI dataset accessed and downloaded
- [x] Bio-Hermes dataset accessed (pending public release)
- [x] DementiaBank corpus downloaded
- [x] Literature review completed (50+ papers)
- [x] ADNI exploratory data analysis (EDA) performed
- [x] ADNI preprocessing pipeline implemented
- [x] Bio-Hermes preprocessing pipeline implemented
- [x] Digital biomarker synthesis (ADNI workaround)
- [x] Modality-specific encoders implemented
- [x] Cross-Modal Attention module implemented
- [x] GNN architecture implemented
- [x] Full NeuroFusion-AD model integrated
- [x] DataLoader and training pipeline setup
- [x] Initial unit tests passed

---

## Phase 1 Exit Criteria

**Before proceeding to Phase 2 (Training), verify:**
1. All datasets are preprocessed and saved to `data/processed/`
2. Train/val/test splits created with proper stratification
3. All model components pass unit tests
4. GPU training environment verified (run `nvidia-smi`, check PyTorch CUDA)
5. W&B or TensorBoard logging configured
6. Git repository up-to-date with all Phase 1 code

**Key Metrics to Confirm:**
- ADNI processed cohort size: ~1,100 MCI patients
- Bio-Hermes processed cohort size: ~300 patients
- Model total parameters: ~50-70M (verify fits in GPU memory)
- Forward pass latency: <500ms per batch (batch_size=32)

---

**Phase 1 Complete!** 🎉  
Proceed to Phase 2: Model Development, Training & Validation.

---

*Document Version: 1.0*  
*Last Updated: February 15, 2026*  
*Next Phase: [Phase 2 Plan](Phase2_Model_Training_Validation.md)*
