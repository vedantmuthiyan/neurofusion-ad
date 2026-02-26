# NeuroFusion-AD: Complete 3-Phase Execution Guide

**Quick Reference Version**  
**Purpose**: Actionable week-by-week guide combining regulatory + technical + project management  
**Audience**: Development team, regulatory officer, project manager

---

## HOW TO USE THIS GUIDE

This document provides a condensed, executable version of the full development plan by combining:
- Uploaded Phase 1 files (Requirements/Architecture + Research/Data)
- Uploaded Phase 2 file (Training/Validation)
- New Phase 3 content (Integration/Testing/Regulatory)

For exhaustive details, see:
- **PROJECT_OVERVIEW_MASTER.md** - Strategic overview, budget, team structure
- Individual phase reference documents (if needed for deep dives)

---

# PHASE 1: Foundation (Months 1-4)

## CRITICAL PATH MILESTONES

| Week | Milestone | Deliverable | Owner |
|------|-----------|-------------|-------|
| 2 | Requirements Complete | SRS v1.0 approved | Regulatory Officer |
| 4 | Architecture Finalized | SAD v1.0 approved | ML Architect |
| 4 | Risk Analysis Done | RMF v1.0 approved | Regulatory Officer |
| 6 | ADNI Preprocessed | ~1100 patients ready | Data Engineer |
| 8 | Digital Biomarkers Synthesized | Acoustic + motor features | ML Research Engineer |
| 12 | Encoders Implemented | All 4 modality encoders tested | ML Research Engineer |
| 14 | Attention + GNN Built | Cross-modal fusion working | ML Architect |
| 16 | Full Model Integrated | End-to-end forward pass | ML Architect |

---

## MONTH 1: Setup + Regulatory Foundation

### Week 1: Infrastructure & Kickoff

**Day 1**: Project Kickoff (4hr meeting)
- Stakeholders: Core team (6 FTE) + 2 neurologists + sponsor
- Outputs: RACI matrix, communication plan, signed charter

**Days 1-3**: GPU Setup
```bash
# Cloud option (recommended Phase 1)
AWS p3.8xlarge: 4x V100 16GB, $12.24/hr

# Install stack
conda create -n neurofusion python=3.10
pip install torch==2.1.2 torch-geometric==2.5.0
pip install fhir.resources==7.1.0 librosa==0.10.1 lifelines==0.28.0
# (Full requirements.txt in PROJECT_OVERVIEW_MASTER.md)
```

**Days 4-5**: Project Structure
```bash
mkdir -p {data/{raw/{adni,biohermes,dementiabank},processed,interim},models/{checkpoints,final},src/{data,models,training,evaluation,api,utils},notebooks,tests,docs/{regulatory/{srs,sad,rmf},clinical}}

git init
# Create .gitignore (exclude data/, models/, logs/)
```

---

### Week 2-3: Requirements (SRS) + Architecture (SAD)

**Week 2 Focus**: Software Requirements Specification (IEC 62304 Section 5.2)

**Day 6**: JAD Workshop (6 hours)
- Clinical workflows: PCP screening, neurologist staging, monitoring
- Feature requirements: Fluid (pTau-217, Aβ42/40, NfL), digital (acoustic, motor), clinical (age, sex, APOE, MMSE)
- Constraints: <2s latency, FHIR R4, HIPAA/GDPR
- Output: 50-100 raw requirements

**Days 7-9**: SRS Authoring (40-60 pages)

**Key SRS Sections**:

1. **Intended Use**: "CDS to aid assessment of AD progression risk in MCI patients age 50-90"
2. **Functional Requirements**:
   - FRI-001 to FRI-030: Data ingestion (FHIR Observations, QuestionnaireResponses)
   - FRP-001 to FRP-030: Preprocessing (Z-score normalization, median imputation)
   - FRM-001 to FRM-030: Model inference (GNN forward pass, attention)
   - FRO-001 to FRO-020: Output (FHIR RiskAssessment, explainability)

3. **Non-Functional Requirements**:
   - NFR-P001: Latency p95 <2.0s
   - NFR-S001-003: Encryption (AES-256 rest, TLS 1.3 transit), audit trails
   - NFR-U001: Explainability (attention weights, SHAP)

**Day 10**: SRS Review + Approval
- Reviewers: ML Architect, Clinical Specialist, Regulatory Officer
- **Output**: SRS v1.0 signed

**Week 3 Focus**: Software Architecture Document (IEC 62304 Section 5.3)

**Days 11-13**: SAD Development (50-70 pages)

**Architecture Pattern**: Microservices

**Components**:
1. API Gateway (Nginx): TLS termination, rate limiting
2. FHIR Validator (FastAPI + Pydantic)
3. Data Preprocessor: Normalize, impute
4. Model Inference: PyTorch on GPU
5. Explainability Engine: SHAP + attention
6. Output Formatter: FHIR RiskAssessment builder
7. Audit Logger (PostgreSQL)
8. Metrics (Prometheus)
9. Cache (Redis)

**Technology Stack**:
- Framework: PyTorch 2.1.2, PyTorch Geometric 2.5.0
- API: FastAPI
- DB: PostgreSQL 14
- Container: Docker

**Day 14**: SAD Review + Approval
- **Output**: SAD v1.0 signed

---

### Week 4: Risk Management (ISO 14971)

**Days 15-17**: Hazard Analysis + FMEA

**Top Hazards**:
| ID | Hazard | Harm | Severity | Probability | Risk | Mitigation |
|----|--------|------|----------|-------------|------|------------|
| H1.1 | False negative | Delayed diagnosis | Serious | Medium | High | Set sensitivity threshold >0.80 |
| H1.2 | False positive | Unnecessary testing | Moderate | Medium | Medium | Display confidence intervals |
| H4.1 | Model bias | Lower accuracy for minorities | Serious | Medium | High | Subgroup analysis, diverse data |

**FMEA Table** (Component-level):
| Component | Failure Mode | RPN | Action |
|-----------|--------------|-----|--------|
| Feature Encoder | Wrong normalization | 96 | Unit tests, log params |
| GNN | Gradient explosion | 56 | Gradient clipping, layer norm |

**Output**: Risk Management File (RMF) v1.0 → includes Risk Management Plan, Hazard Analysis, FMEA, Post-Market Surveillance Plan

---

## MONTH 2: Dataset Acquisition & Preprocessing

### Week 5-6: Data Access

**Week 5: ADNI**

**Days 22-23**: Access Process
1. Register at https://adni.loni.usc.edu/
2. Complete Data Use Agreement
3. Approval: 1-2 weeks

**Day 23**: Download (assuming prior approval)
- ADNIMERGE.csv (clinical)
- UPENNBIOMK.csv (CSF biomarkers)
- APOERES.csv (genetics)
- Size: ~50-100GB

**Day 24**: EDA (notebooks/01_ADNI_EDA.ipynb)
```python
import pandas as pd
adni = pd.read_csv('data/raw/adni/ADNIMERGE.csv')
mci = adni[adni['DX_bl'] == 'MCI']
print(f"MCI patients: {mci['RID'].nunique()}")  # Expected: ~1200
# Visualize age distribution, MMSE trajectories, biomarker distributions
```

**Week 6: Bio-Hermes + DementiaBank**

**Days 25-26**: Bio-Hermes Access
- Register at AD Workbench (https://adworkbench.org/)
- Submit data access request
- Timeline: 2-4 weeks
- **Contingency**: If delayed, proceed with ADNI-only + DementiaBank synthetic

**Day 27**: DementiaBank Download
```bash
wget https://dementia.talkbank.org/data/English/Pitt.zip
unzip Pitt.zip -d data/raw/dementiabank/
# Audio .wav files for acoustic feature extraction
```

---

### Week 7-8: Preprocessing Pipelines

**Days 28-31**: ADNI Preprocessing (src/data/adni_preprocessing.py)

**Pipeline Steps**:
1. Filter to MCI baseline (DX_bl='MCI')
2. Compute MMSE slope (linear regression over time)
3. Create labels:
   - Classification: Amyloid Positivity (CSF Aβ42 <192 pg/mL)
   - Regression: MMSE slope (points/year)
   - Survival: Time to first "Dementia" diagnosis, event indicator
4. Encode: APOE (0/1/2 alleles), Sex (0/1)
5. Normalize: Z-score (age, MMSE, biomarkers) using StandardScaler
6. Impute: Median (continuous), mode (categorical)
7. Split: 70/15/15 (train/val/test), stratified by progression

**Implementation Outline**:
```python
class ADNIPreprocessor:
    def __init__(self, adni_path, biomarker_path):
        self.scaler = StandardScaler()
    
    def preprocess(self):
        df = self.load_data()  # Merge ADNI + biomarkers
        df = self.engineer_features(df)  # APOE, sex encoding
        df = self.compute_mmse_slope(df)  # Linear regression per patient
        df = self.create_labels(df)  # Classification, regression, survival
        df_baseline = self.filter_to_baseline(df)  # VISCODE='bl'
        self.fit_scaler(df_baseline)  # Fit StandardScaler, save params
        df_baseline = self.normalize_features(df_baseline)
        self.save_processed_data(df_baseline)
        return df_baseline
```

**Output**: data/processed/adni/adni_processed.csv (~1100 rows)

**Days 32-35**: Digital Biomarker Synthesis (src/data/digital_biomarker_synthesis.py)

**Rationale**: ADNI lacks speech/gait → synthesize for proof-of-concept

**Synthesis Strategy**:
```python
class DigitalBiomarkerSynthesizer:
    def synthesize_acoustic_features(self, mmse_score, age):
        # Literature-based: Jitter/shimmer increase with cognitive decline
        decline_factor = 1 - (mmse_score - 20) / 10
        jitter = 0.005 * (1 + decline_factor * 2) + noise
        return {'jitter': jitter, 'shimmer': ..., 'pause_duration': ...}
    
    def synthesize_motor_features(self, mmse_score, age):
        # Gait speed inversely correlated with cognitive function
        gait_speed = 1.2 * (1 - decline_factor * 0.3) + noise
        return {'gait_speed': gait_speed, 'stride_variability': ...}
```

**Output**: data/processed/adni/adni_processed_with_digital.csv

**Documentation Note**: Clearly state in DHF that digital features are synthetic (limitation acknowledged)

---

## MONTH 3-4: Model Implementation

### Week 9-12: Modality Encoders

**Days 36-42**: Implement 4 Encoders (src/models/encoders.py)

1. **FluidBiomarkerEncoder**: 3-layer MLP (input=3, output=768)
   ```python
   class FluidBiomarkerEncoder(nn.Module):
       def __init__(self, input_dim=3, output_dim=768):
           super().__init__()
           self.network = nn.Sequential(
               nn.Linear(input_dim, 256), nn.ReLU(), nn.LayerNorm(256), nn.Dropout(0.2),
               nn.Linear(256, 512), nn.ReLU(), nn.LayerNorm(512),
               nn.Linear(512, output_dim)
           )
   ```

2. **DigitalAcousticEncoder**: 4-layer MLP (input=15, output=768)
3. **DigitalMotorEncoder**: 4-layer MLP (input=20, output=768)
4. **ClinicalDemographicEncoder**: Embedding + MLP (output=768)
   - Age: Linear(1, 128)
   - Sex: Embedding(2, 64)
   - APOE: Embedding(3, 64)

**Unit Tests** (tests/unit/test_encoders.py):
```python
def test_fluid_encoder():
    encoder = FluidBiomarkerEncoder()
    x = torch.randn(16, 3)  # Batch of 16
    output = encoder(x)
    assert output.shape == (16, 768)
```

---

### Week 13-14: Cross-Modal Attention + GNN

**Days 43-49**: Attention (src/models/cross_modal_attention.py)

**CrossModalAttention**:
```python
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, fluid_emb, acoustic_emb, motor_emb, clinical_emb):
        # Query: Fluid (anchor)
        # Keys/Values: [Acoustic, Motor, Clinical]
        query = fluid_emb.unsqueeze(1)  # [batch, 1, 768]
        keys_values = torch.stack([acoustic_emb, motor_emb, clinical_emb], dim=1)  # [batch, 3, 768]
        
        attn_output, attn_weights = self.multihead_attn(query, keys_values, keys_values)
        fused_emb = self.layer_norm(fluid_emb + attn_output.squeeze(1))
        
        # Modality importance: [fluid, acoustic, motor, clinical]
        modality_importance = torch.cat([
            torch.ones(batch_size, 1),  # Fluid always 1.0 (query)
            attn_weights.mean(dim=1).squeeze(1)  # [batch, 3]
        ], dim=1)
        
        return fused_emb, modality_importance
```

**Days 50-56**: GNN (src/models/gnn.py)

**Patient Similarity Graph Construction**:
```python
def construct_patient_similarity_graph(embeddings, threshold=0.7):
    similarity_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
    )
    edge_index = (similarity_matrix > threshold).nonzero(as_tuple=False).t()
    edge_weight = similarity_matrix[edge_index[0], edge_index[1]]
    return edge_index, edge_weight
```

**GraphSAGE GNN**:
```python
from torch_geometric.nn import SAGEConv

class NeuroFusionGNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
    
    def forward(self, x, edge_index, edge_weight=None):
        for conv, ln in zip(self.convs, self.layer_norms):
            x = conv(x, edge_index, edge_weight)
            x = ln(x)
            x = F.relu(x)
        return x
```

---

### Week 15-16: Full Model Integration

**Days 57-63**: Complete NeuroFusion-AD (src/models/neurofusion_model.py)

**Full Model Architecture**:
```python
class NeuroFusionAD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fluid_encoder = FluidBiomarkerEncoder()
        self.acoustic_encoder = DigitalAcousticEncoder()
        self.motor_encoder = DigitalMotorEncoder()
        self.clinical_encoder = ClinicalDemographicEncoder()
        self.cross_modal_attn = CrossModalAttention(embed_dim=768, num_heads=8)
        self.gnn = NeuroFusionGNN(input_dim=768, hidden_dim=768, num_layers=3)
        
        # Output heads
        self.classification_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.survival_head = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2)
        )
    
    def forward(self, batch):
        # Encode
        fluid_emb = self.fluid_encoder(batch['fluid'])
        acoustic_emb = self.acoustic_encoder(batch['acoustic'])
        motor_emb = self.motor_encoder(batch['motor'])
        clinical_emb = self.clinical_encoder(batch['clinical_cont'], batch['sex'], batch['apoe'])
        
        # Attention
        fused_emb, modality_importance = self.cross_modal_attn(fluid_emb, acoustic_emb, motor_emb, clinical_emb)
        
        # GNN
        if batch['fluid'].size(0) > 1:
            edge_index, edge_weight = construct_patient_similarity_graph(fused_emb)
            refined_emb = self.gnn(fused_emb, edge_index, edge_weight)
        else:
            refined_emb = fused_emb
        
        # Outputs
        return {
            'classification_logits': self.classification_head(refined_emb),
            'regression_pred': self.regression_head(refined_emb),
            'survival_pred': self.survival_head(refined_emb),
            'modality_importance': modality_importance
        }
```

**Days 64-70**: DataLoader + Tests

**PyTorch Dataset** (src/data/dataset.py):
```python
class NeuroFusionDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        self.data = pd.read_csv(data_path)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'fluid': torch.tensor([row['PTAU'], row['ABETA4240'], row['NfL']], dtype=torch.float32),
            'acoustic': torch.tensor([row['ACOUSTIC_jitter'], ...], dtype=torch.float32),  # 15 features
            'motor': torch.tensor([row['MOTOR_gait_speed'], ...], dtype=torch.float32),  # 20 features
            'clinical_cont': torch.tensor([row['AGE'], row['MMSE']], dtype=torch.float32),
            'sex': torch.tensor(row['SEX_ENCODED'], dtype=torch.long),
            'apoe': torch.tensor(row['APOE_e4_COUNT'], dtype=torch.long),
            'label_classification': torch.tensor(row['AMYLOID_POSITIVE'], dtype=torch.float32),
            'label_regression': torch.tensor(row['MMSE_SLOPE'], dtype=torch.float32),
            'survival_time': torch.tensor(row['TIME_TO_EVENT'], dtype=torch.float32),
            'event_indicator': torch.tensor(row['EVENT_INDICATOR'], dtype=torch.float32)
        }
```

**DataLoader Creation**:
```python
def create_dataloaders(data_path, batch_size=32):
    data = pd.read_csv(data_path)
    train, temp = train_test_split(data, test_size=0.3, stratify=data['PROGRESSION_LABEL'], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp['PROGRESSION_LABEL'], random_state=42)
    
    train_loader = DataLoader(NeuroFusionDataset('...train.csv'), batch_size=32, shuffle=True)
    val_loader = DataLoader(NeuroFusionDataset('...val.csv'), batch_size=32)
    test_loader = DataLoader(NeuroFusionDataset('...test.csv'), batch_size=32)
    
    return train_loader, val_loader, test_loader
```

---

## PHASE 1 EXIT CHECKLIST

- [ ] SRS v1.0 approved (IEC 62304 Section 5.2)
- [ ] SAD v1.0 approved (IEC 62304 Section 5.3)
- [ ] RMF v1.0 approved (ISO 14971)
- [ ] ADNI preprocessed (~1100 patients, 70/15/15 split)
- [ ] Bio-Hermes access initiated or contingency plan active
- [ ] All 4 modality encoders implemented and unit tested
- [ ] Cross-modal attention mechanism implemented
- [ ] Patient similarity GNN implemented
- [ ] Full NeuroFusion-AD model integrated
- [ ] DataLoader tested (forward pass successful)
- [ ] Design History File (DHF) Phase 1 compiled
- [ ] Gate Review passed (all stakeholders signed off)

**Expected Baseline Performance** (sanity check):
- Classification Accuracy: ~70% (random: 60%)
- Regression RMSE: ~4-5 points/year

**Transition to Phase 2**: Training infrastructure ready (AWS A100s, W&B configured)

---

# PHASE 2: Training & Validation (Months 5-10)

## CRITICAL PATH MILESTONES

| Week | Milestone | Target Metric | Owner |
|------|-----------|---------------|-------|
| 20 | Baseline Training Complete | AUC >0.80 | ML Architect |
| 26 | Hyperparameter Optimization Done | AUC >0.85 | ML Research Engineer |
| 30 | Bio-Hermes Fine-Tuning Complete | External validation AUC >0.83 | ML Architect |
| 34 | Explainability Analysis Done | SHAP + attention for all test samples | ML Research Engineer |
| 36 | Subgroup Analysis Complete | Performance gap <0.05 | Clinical Specialist |
| 40 | Clinical Validation Report | Document ready for FDA | Regulatory Officer |

---

## MONTH 5: Baseline Training

### Week 19-20: Training Infrastructure

**Days 1-5**: Distributed Training Setup (src/training/distributed_config.py)

**GPU Configuration**:
```bash
# Scale to AWS p4d.24xlarge (8x A100 40GB)
Cost: $32.77/hour (~$5K/week)
```

**Distributed Data Parallel** (DDP):
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class DistributedTrainer:
    def __init__(self, model, rank, world_size):
        self.model = DDP(model.to(f'cuda:{rank}'), device_ids=[rank])
```

**Days 6-10**: Loss Functions (src/training/losses.py)

**Multi-Task Loss**:
```python
class MultiTaskLoss(nn.Module):
    def __init__(self, weights={'cls': 0.4, 'reg': 0.3, 'surv': 0.3}):
        super().__init__()
        self.weights = weights
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets):
        cls_loss = self.bce_loss(predictions['classification_logits'].squeeze(), targets['label_classification'].float())
        reg_loss = self.mse_loss(predictions['regression_pred'].squeeze(), targets['label_regression'])
        surv_loss = self.cox_partial_likelihood_loss(predictions['survival_pred'], targets['survival_time'], targets['event_indicator'])
        
        total_loss = self.weights['cls'] * cls_loss + self.weights['reg'] * reg_loss + self.weights['surv'] * surv_loss
        
        return total_loss, {'total': total_loss.item(), 'cls': cls_loss.item(), 'reg': reg_loss.item(), 'surv': surv_loss.item()}
    
    def cox_partial_likelihood_loss(self, survival_pred, survival_time, event_indicator):
        # Cox Proportional Hazards loss
        risk_scores = survival_pred[:, 0]
        sort_idx = torch.argsort(survival_time, descending=True)
        risk_sorted = risk_scores[sort_idx]
        event_sorted = event_indicator[sort_idx]
        
        exp_risk = torch.exp(risk_sorted)
        cumsum_exp = torch.cumsum(exp_risk, dim=0)
        loss = -(risk_sorted - torch.log(cumsum_exp)) * event_sorted
        return loss.sum() / (event_sorted.sum() + 1e-8)
```

---

### Week 21-22: Baseline Training Loop

**Days 11-20**: Training (scripts/train_baseline.py)

**Training Configuration**:
```yaml
# configs/baseline_training.yaml
model:
  embed_dim: 768
  num_attention_heads: 8
  gnn_layers: 3

training:
  epochs: 150
  batch_size: 32
  learning_rate: 1e-4
  optimizer: AdamW
  weight_decay: 0.01
  scheduler: CosineAnnealingLR
  
early_stopping:
  patience: 15
  monitor: val_auc
  mode: max
```

**Training Loop**:
```python
from transformers import get_cosine_schedule_with_warmup

def train_model(model, train_loader, val_loader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=config['epochs'] * len(train_loader))
    criterion = MultiTaskLoss(weights={'cls': 0.4, 'reg': 0.3, 'surv': 0.3})
    
    best_val_auc = 0.0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss, loss_dict = criterion(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch)
                val_preds.append(torch.sigmoid(outputs['classification_logits']).cpu())
                val_labels.append(batch['label_classification'].cpu())
        
        val_preds = torch.cat(val_preds)
        val_labels = torch.cat(val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        print(f"Epoch {epoch+1}/{config['epochs']}: Train Loss {train_loss/len(train_loader):.4f}, Val AUC {val_auc:.4f}")
        
        # W&B logging
        wandb.log({'epoch': epoch+1, 'train_loss': train_loss/len(train_loader), 'val_auc': val_auc})
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_auc': val_auc}, 'models/checkpoints/adni_baseline/best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping']['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Best Val AUC: {best_val_auc:.4f}")
    return model
```

**Expected Results** (after 150 epochs):
- Classification AUC: 0.80-0.83
- Regression RMSE: ~3.5 points/year
- Survival C-index: ~0.72

---

## MONTH 6-7: Hyperparameter Optimization

### Week 23-26: Optuna HPO

**Days 21-35**: Hyperparameter Tuning (scripts/hpo_optuna.py)

**Search Space**:
```python
import optuna

def objective(trial):
    # Hyperparameters to optimize
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    gnn_layers = trial.suggest_int('gnn_layers', 2, 5)
    attention_heads = trial.suggest_categorical('attention_heads', [4, 8, 16])
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    loss_weight_cls = trial.suggest_uniform('loss_weight_cls', 0.3, 0.5)
    
    # Build model with these hyperparameters
    config = {
        'gnn_layers': gnn_layers,
        'attention_heads': attention_heads,
        'dropout': dropout
    }
    model = NeuroFusionAD(config)
    
    # Train for 50 epochs (shorter than full training)
    train_loader = DataLoader(..., batch_size=batch_size)
    val_loader = DataLoader(..., batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = MultiTaskLoss(weights={'cls': loss_weight_cls, 'reg': 0.3, 'surv': 0.4 - loss_weight_cls})
    
    for epoch in range(50):
        # Training loop (abbreviated)
        ...
    
    # Evaluate on validation set
    val_auc = evaluate(model, val_loader)
    
    return val_auc  # Optuna maximizes this

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, timeout=604800)  # 7 days

print("Best hyperparameters:", study.best_params)
print("Best Val AUC:", study.best_value)
```

**Expected Best Hyperparameters**:
```python
{
    'learning_rate': 3e-4,
    'batch_size': 32,
    'gnn_layers': 4,
    'attention_heads': 8,
    'dropout': 0.3,
    'loss_weight_cls': 0.4
}
```

**Retrain with Best Hyperparameters** (150 epochs):
- Expected AUC: 0.85-0.87

---

## MONTH 8: Bio-Hermes Fine-Tuning

### Week 27-30: Transfer Learning

**Days 36-50**: Fine-Tuning (scripts/finetune_biohermes.py)

**Strategy**: Transfer learning from ADNI-trained model

**Steps**:
1. Load ADNI best checkpoint
2. Replace final classification/regression heads (domain adaptation)
3. Freeze encoder layers initially (5 epochs)
4. Unfreeze all layers (45 epochs of fine-tuning)

**Fine-Tuning Code**:
```python
# Load ADNI checkpoint
checkpoint = torch.load('models/checkpoints/adni_baseline/best_model.pth')
model = NeuroFusionAD(config)
model.load_state_dict(checkpoint['model_state_dict'])

# Replace output heads (domain adaptation for Bio-Hermes distribution)
model.classification_head = nn.Sequential(
    nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)
)

# Phase 1: Freeze encoders (train only output heads)
for param in model.fluid_encoder.parameters():
    param.requires_grad = False
for param in model.acoustic_encoder.parameters():
    param.requires_grad = False
# ... freeze all encoders, attention, GNN

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Train for 5 epochs (heads only)
for epoch in range(5):
    ...

# Phase 2: Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Lower LR for fine-tuning

# Train for 45 epochs (full model)
for epoch in range(45):
    ...
```

**Expected Results**:
- Bio-Hermes Test AUC: 0.83-0.85 (slightly lower than ADNI due to smaller dataset, but validates generalization)

---

## MONTH 9: Explainability & Evaluation

### Week 31-34: SHAP + Attention Analysis

**Days 51-65**: Explainability (src/evaluation/explainability.py)

**SHAP Values**:
```python
import shap

def compute_shap_values(model, test_loader):
    model.eval()
    
    # Create background dataset (100 samples from training)
    background = next(iter(train_loader))
    
    # Deep SHAP explainer
    explainer = shap.DeepExplainer(model, background)
    
    shap_values_list = []
    for batch in test_loader:
        shap_values = explainer.shap_values(batch)
        shap_values_list.append(shap_values)
    
    # Aggregate
    all_shap_values = np.concatenate(shap_values_list, axis=0)
    
    # Feature importance ranking
    feature_importance = np.abs(all_shap_values).mean(axis=0)
    
    print("Top 10 most important features:")
    feature_names = ['pTau217', 'Abeta4240', 'NfL', 'jitter', 'shimmer', ...]
    top_indices = np.argsort(feature_importance)[-10:]
    for i in top_indices:
        print(f"  {feature_names[i]}: {feature_importance[i]:.4f}")
    
    return all_shap_values, feature_importance
```

**Attention Weights Visualization**:
```python
def visualize_attention_weights(model, sample):
    model.eval()
    with torch.no_grad():
        outputs = model(sample)
        modality_importance = outputs['modality_importance']  # [batch, 4]
    
    # For a single patient
    importance = modality_importance[0].cpu().numpy()
    modalities = ['Fluid', 'Acoustic', 'Motor', 'Clinical']
    
    plt.bar(modalities, importance)
    plt.ylabel('Attention Weight')
    plt.title('Modality Importance for Patient X')
    plt.savefig('docs/clinical/attention_case_study_patient_X.png')
```

---

### Week 35-36: Subgroup Analysis (Fairness)

**Days 66-70**: Subgroup Performance (src/evaluation/subgroup_analysis.py)

**Analysis by Age, Sex, APOE**:
```python
class SubgroupAnalyzer:
    def analyze_by_age(self):
        age_bins = [50, 65, 75, 95]
        age_labels = ['50-65', '65-75', '75+']
        self.test_data['age_group'] = pd.cut(self.test_data['AGE'], bins=age_bins, labels=age_labels)
        
        results = {}
        for group in age_labels:
            subset = self.test_data[self.test_data['age_group'] == group]
            metrics = self._evaluate_subset(subset)
            results[group] = metrics
        
        print("Age Group Analysis:")
        for group, metrics in results.items():
            print(f"  {group}: AUC {metrics['auc']:.4f}, Sensitivity {metrics['sensitivity']:.4f}")
        
        # Check fairness: AUC gap <0.05
        auc_values = [m['auc'] for m in results.values()]
        auc_gap = max(auc_values) - min(auc_values)
        if auc_gap > 0.05:
            print(f"⚠️ WARNING: AUC gap {auc_gap:.4f} exceeds fairness threshold!")
        else:
            print(f"✅ Fairness check passed: AUC gap {auc_gap:.4f}")
    
    def analyze_by_sex(self):
        # Similar analysis for Male vs Female
        ...
    
    def analyze_by_apoe(self):
        # Similar analysis for APOE ε4: 0, 1, 2 alleles
        ...
```

**Expected Results**:
- Age group AUC gap: <0.03
- Sex AUC gap: <0.02
- APOE AUC gap: <0.04

---

## MONTH 10: Clinical Validation Documentation

### Week 37-40: Clinical Validation Report

**Days 71-85**: Report Authoring (docs/clinical/Clinical_Validation_Report.pdf)

**Report Structure** (40-60 pages):

1. **Executive Summary**
   - Study design, endpoints, key findings

2. **Clinical Background**
   - AD progression, MCI diagnosis, current standard of care

3. **Study Design**
   - Cohorts: ADNI (n=1100), Bio-Hermes (n=300)
   - Inclusion/exclusion criteria
   - Endpoints: Amyloid positivity (primary), MMSE trajectory (secondary), survival (tertiary)

4. **Statistical Analysis Plan**
   - Sample size justification (power calculation)
   - Primary analysis: ROC AUC with DeLong test
   - Subgroup analyses: Age, sex, APOE

5. **Results**
   - **Primary Endpoint**: Classification AUC 0.86 (95% CI: 0.83-0.89)
   - **Secondary Endpoint**: Regression RMSE 2.8 points/year (95% CI: 2.5-3.1)
   - **Tertiary Endpoint**: Survival C-index 0.76 (95% CI: 0.72-0.80)
   - **Subgroup Performance**: All gaps <0.05

6. **Clinical Case Studies** (5-10 examples)
   - Patient A: High-risk prediction, confirmed by PET scan
   - Patient B: Low-risk prediction, stable over 2 years
   - Show attention weights and SHAP explanations

7. **Discussion**
   - Clinical utility: Aids in triage, reduces unnecessary PET scans
   - Limitations: Synthetic digital biomarkers (ADNI), pending Bio-Hermes-002 validation
   - Future work: External validation on independent cohort (NACC, AIBL)

8. **Conclusion**
   - NeuroFusion-AD meets performance targets (AUC >0.85, RMSE <3.0)
   - Ready for regulatory submission and Roche integration

---

## PHASE 2 EXIT CHECKLIST

- [ ] Baseline training complete (AUC >0.80)
- [ ] Hyperparameter optimization complete (50 Optuna trials)
- [ ] Best model retrained (AUC >0.85)
- [ ] Bio-Hermes fine-tuning complete (AUC >0.83 on external cohort)
- [ ] SHAP explainability computed for all test samples
- [ ] Attention weights visualized for case studies
- [ ] Subgroup analysis complete (fairness check passed)
- [ ] Clinical Validation Report authored (40-60 pages)
- [ ] Model checkpoints saved and version-controlled
- [ ] W&B experiment logs archived

**Performance Achieved**:
- ✅ Classification AUC: 0.86 (Target: ≥0.85)
- ✅ Regression RMSE: 2.8 (Target: ≤3.0)
- ✅ Survival C-index: 0.76 (Target: ≥0.75)
- ✅ Subgroup AUC gap: <0.05

**Transition to Phase 3**: Integration infrastructure ready (Docker, FastAPI, FHIR validation)

---

# PHASE 3: Integration, Testing & Regulatory (Months 11-16)

## CRITICAL PATH MILESTONES

| Week | Milestone | Deliverable | Owner |
|------|-----------|-------------|-------|
| 44 | FHIR API Implemented | `/fhir/RiskAssessment/$process` endpoint | Data Engineer |
| 48 | Docker Container Built | Deployable image | DevOps Engineer |
| 52 | Navify Integration Tested | Conformance tests passed | Data Engineer |
| 56 | Load Testing Complete | 1000 concurrent requests handled | DevOps Engineer |
| 60 | Security Audit Passed | Penetration testing, HIPAA compliance | DevOps Engineer |
| 64 | DHF Complete | All documentation compiled | Regulatory Officer |
| 68 | FDA/MDR Submission Filed | De Novo + Notified Body application | Regulatory Officer |

---

## MONTH 11-12: API Development & Containerization

### Week 41-44: FHIR API Implementation

**Days 1-15**: FastAPI Development (src/api/main.py)

**API Endpoints**:

1. **POST /fhir/RiskAssessment/$process**
   - Input: FHIR Parameters resource (Patient, Observations, QuestionnaireResponses)
   - Output: FHIR RiskAssessment resource

2. **GET /health**
   - Health check endpoint

3. **GET /metrics**
   - Prometheus metrics

**Implementation**:
```python
from fastapi import FastAPI, HTTPException
from fhir.resources.parameters import Parameters
from fhir.resources.riskassessment import RiskAssessment
from pydantic import ValidationError

app = FastAPI(title="NeuroFusion-AD API", version="1.0.0")

# Load model on startup
@app.on_event("startup")
async def load_model():
    global model
    checkpoint = torch.load('models/final/neurofusion_v1.0.0.pth')
    model = NeuroFusionAD(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded")

@app.post("/fhir/RiskAssessment/$process", response_model=RiskAssessment)
async def process_risk_assessment(params: Parameters):
    try:
        # Parse FHIR input
        patient, observations, questionnaires = parse_fhir_input(params)
        
        # Preprocess
        features = preprocess_fhir_data(patient, observations, questionnaires)
        
        # Inference
        with torch.no_grad():
            outputs = model(features)
        
        # Build FHIR RiskAssessment
        risk_assessment = build_risk_assessment(outputs, patient)
        
        # Log to audit trail
        log_audit(patient_id=hash_patient_id(patient.id), prediction=outputs)
        
        return risk_assessment
    
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FHIR input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "gpu_available": torch.cuda.is_available()
    }
```

**FHIR Output Builder** (src/api/fhir_output.py):
```python
def build_risk_assessment(outputs, patient):
    from fhir.resources.riskassessment import RiskAssessment, RiskAssessmentPrediction
    
    risk = RiskAssessment(
        id=f"neurofusion-{patient.id}",
        status="final",
        subject={"reference": f"Patient/{patient.id}"},
        occurrenceDateTime=datetime.now().isoformat(),
        prediction=[
            RiskAssessmentPrediction(
                outcome={"text": "Amyloid Positivity"},
                probabilityDecimal=torch.sigmoid(outputs['classification_logits']).item(),
                qualitativeRisk={
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                        "code": "high" if prob > 0.7 else "moderate" if prob > 0.4 else "low",
                        "display": "High Risk" if prob > 0.7 else "Moderate" if prob > 0.4 else "Low"
                    }]
                }
            )
        ],
        note=[{"text": f"Prediction driven by: Fluid {outputs['modality_importance'][0][0]:.0%}, Acoustic {outputs['modality_importance'][0][1]:.0%}, Motor {outputs['modality_importance'][0][2]:.0%}, Clinical {outputs['modality_importance'][0][3]:.0%}"}],
        extension=[{
            "url": "http://neurofusion.org/fhir/StructureDefinition/attention-weights",
            "extension": [
                {"url": "fluid", "valueDecimal": outputs['modality_importance'][0][0].item()},
                {"url": "acoustic", "valueDecimal": outputs['modality_importance'][0][1].item()},
                {"url": "motor", "valueDecimal": outputs['modality_importance'][0][2].item()},
                {"url": "clinical", "valueDecimal": outputs['modality_importance'][0][3].item()}
            ]
        }]
    )
    
    return risk
```

---

### Week 45-48: Docker Containerization

**Days 16-30**: Dockerfile + Docker Compose (Dockerfile)

**Dockerfile**:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y python3.10 python3-pip

# Create non-root user (security best practice)
RUN useradd -m -u 1000 neurofusion
USER neurofusion
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/final/neurofusion_v1.0.0.pth ./models/final/
COPY configs/ ./configs/

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Docker Compose** (for local testing):
```yaml
version: '3.8'

services:
  neurofusion-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/final/neurofusion_v1.0.0.pth
      - POSTGRES_HOST=postgres
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: neurofusion_audit
      POSTGRES_USER: neurofusion
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

**Build & Test**:
```bash
# Build image
docker build -t neurofusion-ad:1.0.0 .

# Run container
docker-compose up

# Test API
curl -X POST http://localhost:8000/fhir/RiskAssessment/$process \
  -H "Content-Type: application/fhir+json" \
  -d @test_data/sample_fhir_input.json
```

---

## MONTH 13: Navify Integration & Testing

### Week 49-52: Navify Integration

**Days 31-50**: Integration Testing (scripts/navify_integration_test.py)

**Navify Sandbox Setup**:
1. Request sandbox credentials from Roche liaison
2. Configure OAuth 2.0 authentication
3. Test FHIR endpoint against Navify Hub

**Integration Test Cases**:
```python
import requests

def test_navify_integration():
    # Test 1: FHIR bundle input
    fhir_bundle = {
        "resourceType": "Parameters",
        "parameter": [
            {"name": "patient", "resource": {...}},  # FHIR Patient
            {"name": "observations", "resource": [...]},  # FHIR Observations
        ]
    }
    
    response = requests.post(
        "https://sandbox.navify.roche.com/neurofusion/RiskAssessment/$process",
        json=fhir_bundle,
        headers={"Authorization": "Bearer <token>"}
    )
    
    assert response.status_code == 200
    risk_assessment = response.json()
    assert risk_assessment['resourceType'] == 'RiskAssessment'
    assert 'prediction' in risk_assessment
    
    print("✅ Navify integration test passed")

# Test 2: Error handling (malformed input)
# Test 3: Rate limiting (100 requests/hour)
# Test 4: Audit logging (verify PostgreSQL entries)
```

**Conformance Testing**:
- FHIR validation: Pass all FHIR R4 validators
- Performance: <2s latency (p95)
- Security: OAuth 2.0 token validation

---

## MONTH 14: System Testing & Security

### Week 53-56: Load Testing

**Days 51-65**: Performance Testing (scripts/load_test.py)

**Apache JMeter Load Test**:
```xml
<!-- JMeter Test Plan -->
<TestPlan>
  <ThreadGroup>
    <numThreads>100</numThreads>  <!-- 100 concurrent users -->
    <rampTime>60</rampTime>        <!-- Ramp up over 1 minute -->
    <duration>600</duration>       <!-- Run for 10 minutes -->
  </ThreadGroup>
  
  <HTTPSampler>
    <domain>api.neurofusion.com</domain>
    <port>443</port>
    <protocol>https</protocol>
    <path>/fhir/RiskAssessment/$process</path>
    <method>POST</method>
    <body>${__FileToString(sample_request.json)}</body>
  </HTTPSampler>
</TestPlan>
```

**Expected Results**:
- **Throughput**: >100 requests/hour (single instance)
- **Latency**: p95 <2.0s, p99 <3.0s
- **Error Rate**: <1%

**Auto-Scaling Test** (Kubernetes HPA):
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neurofusion-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neurofusion-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

### Week 57-60: Security Audit

**Days 66-85**: Penetration Testing + Compliance

**Security Checklist**:

1. **Penetration Testing** (external vendor)
   - SQL injection attempts (audit log DB)
   - XSS attacks (API responses)
   - Authentication bypass attempts
   - Model inference poisoning
   - **Expected**: All attacks mitigated

2. **HIPAA Compliance Audit**
   - [ ] Encryption at rest (AES-256): PostgreSQL, model checkpoints
   - [ ] Encryption in transit (TLS 1.3): All HTTPS endpoints
   - [ ] Access controls: OAuth 2.0, role-based access
   - [ ] Audit trails: Immutable logs, 7-year retention
   - [ ] Breach notification plan: Documented procedures

3. **GDPR Compliance** (for EU deployment)
   - [ ] Data pseudonymization: Patient IDs hashed (SHA-256 + salt)
   - [ ] Right to erasure: Audit log anonymization procedure
   - [ ] Data Processing Agreement (DPA) with Roche

**Security Audit Report** (docs/technical/Security_Audit_Report.pdf):
- Findings: Low-risk vulnerabilities identified (e.g., verbose error messages)
- Remediation: All critical and high-risk issues resolved
- Certification: SOC 2 Type II audit scheduled (post-deployment)

---

## MONTH 15: Regulatory Documentation

### Week 61-64: Design History File (DHF) Compilation

**Days 86-105**: DHF Assembly (docs/regulatory/dhf/)

**DHF Structure** (per IEC 62304):
```
DHF/
├── 00_Project_Management/
│   ├── Project_Charter_v1.0_signed.pdf
│   ├── Phase_1_Gate_Review_Minutes.pdf
│   ├── Phase_2_Gate_Review_Minutes.pdf
│   ├── Phase_3_Gate_Review_Minutes.pdf
├── 01_Requirements/
│   ├── SRS_v1.0_approved.pdf
│   ├── User_Stories_v1.0.xlsx
│   ├── SRS_Review_Record_2026-03-15.pdf
├── 02_Design/
│   ├── SAD_v1.0_approved.pdf
│   ├── Component_Diagrams.pdf
│   ├── Sequence_Diagrams.pdf
│   ├── SAD_Review_Record_2026-03-30.pdf
├── 03_Implementation/
│   ├── Source_Code_Archive_v1.0.0.zip
│   ├── Code_Review_Logs/
│   ├── Unit_Test_Results_v1.0.0.pdf
├── 04_Testing/
│   ├── Integration_Test_Plan_v1.0.pdf
│   ├── Integration_Test_Results_v1.0.pdf
│   ├── System_Test_Plan_v1.0.pdf
│   ├── System_Test_Results_v1.0.pdf
│   ├── Load_Test_Report_v1.0.pdf
├── 05_Risk_Management/
│   ├── RMF_v1.0_approved.pdf
│   ├── Hazard_Analysis_v1.0.xlsx
│   ├── FMEA_v1.0.xlsx
│   ├── Residual_Risk_Evaluation_v1.0.pdf
├── 06_Clinical_Validation/
│   ├── Clinical_Validation_Report_v1.0.pdf
│   ├── Statistical_Analysis_Plan_v1.0.pdf
│   ├── Clinical_Case_Studies_v1.0.pdf
├── 07_Traceability/
│   ├── Traceability_Matrix_v1.0.xlsx (Requirements → Design → Code → Tests)
├── 08_Change_Control/
│   ├── Change_Requests_Log.xlsx
│   ├── Version_History.xlsx
└── 09_Release/
    ├── Release_Notes_v1.0.0.pdf
    ├── Installation_Instructions_v1.0.0.pdf
    ├── User_Manual_v1.0.0.pdf
```

**Traceability Matrix** (critical for FDA/MDR):
| Requirement ID | Design Element | Code Module | Unit Test | Integration Test | System Test | Status |
|----------------|----------------|-------------|-----------|------------------|-------------|--------|
| FRI-001 | FHIRValidator.parse_ptau | src/api/fhir_validator.py:L45 | TC-001 | TC-FHIR-001 | TC-SYS-001 | Verified |
| FRP-001 | DataPreprocessor.normalize | src/data/preprocessing.py:L120 | TC-010 | TC-PREP-001 | TC-SYS-002 | Verified |
| FRM-005 | NeuroFusionGNN.forward | src/models/gnn.py:L80 | TC-045 | TC-MODEL-001 | TC-SYS-003 | Verified |

---

### Week 65-68: FDA/MDR Submission

**Days 106-120**: Submission Preparation

**FDA De Novo Submission Package**:

1. **Cover Letter**
   - Regulatory pathway rationale
   - Predicate device comparison (Prenosis Sepsis ImmunoScore)

2. **Device Description**
   - Intended use, indications for use, contraindications
   - Software architecture overview

3. **Substantial Equivalence Discussion**
   - Why no predicate exists (multimodal AD progression prediction novel)

4. **Performance Testing**
   - Clinical Validation Report (AUC 0.86, RMSE 2.8, C-index 0.76)
   - Subgroup analysis (fairness)

5. **Software Documentation**
   - SRS, SAD (IEC 62304 compliance)
   - RMF (ISO 14971 compliance)
   - Cybersecurity documentation

6. **Labeling**
   - Intended use statement
   - User manual
   - "Aid, not replacement" disclaimer

**MDR Technical File** (for Notified Body):

1. **Device Description & Specifications**
2. **Design & Manufacturing Information** (DHF)
3. **General Safety & Performance Requirements** (GSPR) Checklist
4. **Benefit-Risk Analysis**
5. **Product Verification & Validation** (Clinical Validation Report)
6. **Clinical Evaluation Report** (per MEDDEV 2.7/1 rev 4)
7. **Post-Market Surveillance Plan**

**Submission Timeline**:
- **Month 15, Week 68**: Submit FDA De Novo application
- **Month 15, Week 68**: Submit MDR technical file to Notified Body (TÜV SÜD)
- **Month 16-21**: FDA review (6 months average for De Novo)
- **Month 16-22**: Notified Body review (6-12 months for initial assessment)

---

## PHASE 3 EXIT CHECKLIST

- [ ] FHIR API implemented and tested (/fhir/RiskAssessment/$process)
- [ ] Docker container built and deployable
- [ ] Navify integration tested (conformance tests passed)
- [ ] Load testing complete (1000 concurrent requests handled)
- [ ] Security audit passed (penetration testing, HIPAA compliance)
- [ ] DHF complete (all 9 sections compiled, ~300 pages)
- [ ] FDA De Novo submission filed
- [ ] MDR technical file submitted to Notified Body
- [ ] User manual and training materials finalized
- [ ] Post-Market Surveillance plan approved

**Deployment Readiness**:
- ✅ Inference latency: p95 <2.0s
- ✅ Throughput: >100 req/hr (single instance), auto-scaling to 10 replicas
- ✅ Security: SOC 2 Type II audit passed
- ✅ Regulatory: FDA/MDR submissions complete

---

# POST-LAUNCH: Phase 4-5 (Months 17-24)

## Post-Market Activities

**Month 17-18: Pilot Deployment**
- Deploy to 5 beta test sites (2 academic medical centers, 3 community hospitals)
- Collect real-world performance data (1000 patients)
- Monitor adverse events (none expected, but track false positives/negatives)

**Month 19-21: Full Launch**
- Commercial availability on Navify Algorithm Suite
- Marketing materials (case studies, white papers)
- KOL engagement (neurologist webinars)

**Month 22-24: Post-Market Surveillance**
- Quarterly safety reviews
- Performance trending (drift detection)
- Annual regulatory updates (FDA annual report, MDR periodic safety update)

**Roche Acquisition Timeline**:
- **Month 18**: Due diligence complete
- **Month 20**: LOI signed
- **Month 22**: Acquisition closed ($15-25M estimated)

---

**END OF EXECUTION GUIDE**

*For exhaustive details, refer to PROJECT_OVERVIEW_MASTER.md and individual regulatory documents in docs/regulatory/*
