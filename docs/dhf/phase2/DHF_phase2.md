# NeuroFusion-AD Design History File
## Phase 2 Software Development Record

---

**Document Number:** NFD-DHF-002
**Revision:** 1.0
**Effective Date:** [DATE]
**Product:** NeuroFusion-AD — Multimodal Clinical Decision Support SaMD
**Regulatory Pathway:** FDA De Novo / EU MDR Class IIa
**IEC 62304 Software Safety Class:** Class B
**Applicable Standards:** IEC 62304:2006+AMD1:2015 | ISO 14971:2019 | 21 CFR Part 820 | ISO 13485:2016
**W&B Project IDs:** k58caevv (baseline) | t9s3ngbx (remediated best) | o4pcjy3r (Bio-Hermes fine-tune)

**Prepared by:** Clinical Documentation Specialist, NeuroFusion-AD Program
**Reviewed by:** [Software Engineering Lead] | [Clinical Affairs] | [Regulatory Affairs]
**Approved by:** [Quality Assurance Director]

---

> **PHASE 2B REMEDIATION NOTICE**
> This DHF section documents critical remediation activity arising from discovery of data leakage in the Phase 2A fluid feature set. Specifically, CSF Abeta42 (ABETA42_CSF; Pearson r = −0.864 with amyloid label) was identified as a near-proxy for the prediction target and removed from all training, validation, and test pipelines prior to final model training. All performance figures reported herein reflect the post-remediation (Phase 2B) state. Pre-remediation artefacts are retained in the DHF for traceability per 21 CFR Part 820.181 but are superseded and must not be used operationally. Affected document reference: DRD-001 (Data Requirements Document, Revision 2).

---

## SECTION 1 — PHASE 2 DHF INDEX

### 1.1 Document Hierarchy

```
NFD-DHF-002  Phase 2 Design History File (this document)
├── NFD-SRS-002      Software Requirements Specification, Phase 2
├── NFD-SDD-002      Software Design Description, Phase 2
├── NFD-DRD-001      Data Requirements Document (Rev 2 — post-remediation)
├── NFD-RMF-002      Risk Management File, Phase 2 Additions
├── NFD-VVP-002      Verification & Validation Plan, Phase 2
├── NFD-VVR-002      Verification & Validation Report, Phase 2
├── NFD-TDL-002      Training Decision Log (reproduced in Section 2 below)
├── NFD-MVH-002      Model Version History (reproduced in Section 3 below)
├── NFD-CAL-001      Calibration Report — Temperature Scaling
├── NFD-SUB-001      Subgroup Fairness Analysis Report
├── NFD-PMS-002      Post-Market Surveillance Plan Addendum, Phase 2
└── NFD-REM-001      Remediation Report — ABETA42_CSF Leakage (Phase 2B)
```

### 1.2 Phase 2 Milestone Summary

| Milestone | Reference ID | Status | Notes |
|---|---|---|---|
| Phase 1 Architecture Freeze | NFD-DHF-001 §6 | Complete | v0.1 baseline inherited |
| Data Acquisition & Curation (ADNI) | NFD-DRD-001 Rev 1 | Complete | 494 MCI; 345/74/75 split |
| Phase 2A Baseline Training | W&B: k58caevv | Superseded | Pre-remediation; leakage identified |
| Leakage Identification & Remediation | NFD-REM-001 | Complete | ABETA42_CSF removed |
| Phase 2B HPO (15 trials) | W&B: t9s3ngbx | Complete | Best val AUC = 0.9081 |
| Bio-Hermes-001 Fine-Tuning | W&B: o4pcjy3r | Complete | Encoders frozen |
| Temperature Scaling Calibration | NFD-CAL-001 | Complete | T = 0.76; ECE = 0.0831 |
| ADNI Test Set Evaluation | NFD-VVR-002 §3 | Complete | AUC = 0.890 |
| Bio-Hermes-001 Test Set Evaluation | NFD-VVR-002 §4 | Complete | AUC = 0.907 |
| Subgroup Fairness Review | NFD-SUB-001 | Complete — FLAG | fairness_pass = False |
| Phase 2 DHF Closure | NFD-DHF-002 | In Review | Pending QA sign-off |

### 1.3 Configuration Items Under Change Control

| Item | Identifier | Version | Location |
|---|---|---|---|
| Model weights (ADNI baseline) | NFD-MW-001 | v0.2 | Secure artifact registry |
| Model weights (HPO best) | NFD-MW-002 | v0.3 | W&B: t9s3ngbx |
| Model weights (production) | NFD-MW-003 | v1.0 | W&B: o4pcjy3r |
| Inference codebase | NFD-SRC-002 | 2.0.0 | Git SHA [COMMIT-HASH] |
| Feature preprocessing pipeline | NFD-SRC-003 | 2.1.0 | Git SHA [COMMIT-HASH] |
| Calibration parameters | NFD-CAL-001 | 1.0 | T=0.76, stored in model config |

---

## SECTION 2 — TRAINING DECISION LOG

*Format: Each decision record follows the structure required by IEC 62304 §5.1.1 (software development planning), §5.1.6 (software configuration and change management), and ISO 14971:2019 §8 (risk control measures). Records are numbered sequentially as TDL-2B-XXX. Pre-remediation decisions from Phase 2A are retained as TDL-2A-XXX and marked SUPERSEDED where applicable.*

---

### TDL-2B-001 — CRITICAL REMEDIATION: Removal of ABETA42_CSF from Fluid Feature Set

**Decision Made:**
CSF Abeta42 (ABETA42_CSF) was permanently removed from the fluid biomarker feature vector prior to any Phase 2B training activity. The fluid feature set was reduced from six features to two: [PTAU217, NFL_PLASMA]. All Phase 2A model weights and evaluation artefacts were archived as superseded. No Phase 2A outputs may be used in clinical or regulatory submissions.

**Rationale:**
During Phase 2A internal review, statistical analysis of the ADNI training split revealed Pearson r = −0.864 between ABETA42_CSF and the amyloid binary label. This correlation magnitude indicates that the feature is functionally a proxy for the ground-truth label, constituting severe data leakage. A model trained with access to this feature would learn to predict from the label itself rather than from clinically independent prognostic signals. Deployed performance would catastrophically diverge from training performance when the feature is unavailable or when the assay differs, exposing patients to incorrectly calibrated risk estimates. This failure mode meets the ISO 14971:2019 definition of a hazardous situation (§3.13) with potential for patient harm via inappropriate clinical pathway decisions.

Additionally, the intended use of NeuroFusion-AD specifies plasma-based biomarker inputs for scalable, non-invasive risk assessment. ABETA42_CSF requires lumbar puncture, which is inconsistent with the target clinical workflow on the Navify Algorithm Suite and inconsistent with the two-feature fluid panel available in Bio-Hermes-001 (plasma pTau217 + NfL).

**Alternatives Considered:**
1. *Retain ABETA42_CSF as a conditional input (present/absent flag):* Rejected. Conditional architecture complexity does not eliminate the leakage risk in training; model would continue to preferentially weight the feature when available, producing biased population-level performance estimates.
2. *Apply feature noise augmentation to reduce correlation:* Rejected. Artificially degrading a real biomarker signal to suppress its informativeness is methodologically unsound and would not satisfy FDA guidance on predetermined change control plans.
3. *Stratify training to balance leakage:* Rejected. Leakage is not a sampling imbalance; it cannot be corrected by stratification.
4. *Proceed to submission with 2A weights, disclosing leakage:* Rejected on patient safety and regulatory integrity grounds.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.1.1 (risk management integrated into software planning), §7.1.1 (problem resolution process triggered)
- *ISO 14971 Reference:* Hazard H-007 (see Risk Register §5); Harm: clinician acts on inflated risk estimate → inappropriate treatment initiation or unnecessary invasive diagnostic workup; Probability of Harm (before remediation): High; Severity: Serious; Risk (before remediation): Unacceptable. Post-remediation risk: Acceptable with residual controls (see NFD-RMF-002 §4.3).
- *Residual risk after removal:* The remaining fluid features (PTAU217 proxy via pTau181 in ADNI; NFL_PLASMA) retain known clinical association with amyloid pathology but at correlation magnitudes consistent with legitimate biomarker utility (r values not reported as leakage-threshold; to be confirmed in DRD-001 Rev 2 statistical annex).

---

### TDL-2B-002 — Fluid Feature Assay Mismatch: pTau181 as Proxy for pTau217 in ADNI

**Decision Made:**
Proceed with ADNI training using CSF pTau181 as a proxy input for the PTAU217 model feature slot, with mandatory documentation in all validation reports and the IFU. A limitation statement is embedded in model output metadata at inference time when the ADNI-derived calibration is active.

**Rationale:**
ADNI does not include plasma pTau217 (Roche Elecsys assay), which is the intended clinical input per the product specification. CSF pTau181 is the nearest available surrogate with established, peer-reviewed correlation to amyloid pathology. The Bio-Hermes-001 dataset provides ground-truth plasma pTau217 coverage and is used for fine-tuning and primary external validation. The ADNI model therefore serves as a pre-trained initialization; Bio-Hermes-001 fine-tuning with frozen encoders adjusts the classification layer to the correct assay's distributional characteristics.

**Alternatives Considered:**
1. *Exclude PTAU217 from ADNI training entirely:* Rejected. Removing the feature slot during pre-training and inserting it at fine-tuning stage would require architectural surgery and introduce initialization variance; the proxy approach preserves feature-slot alignment.
2. *Use a cross-assay normalization transform:* Considered. Literature-based linear scaling between pTau181 and pTau217 ranges exists, but introduces an additional unvalidated transformation step with its own uncertainty. Proxy-with-disclosure is preferred as the more transparent approach.
3. *Train exclusively on Bio-Hermes-001:* Rejected. Bio-Hermes-001 sample size (N=661 train) is insufficient for full multimodal model pre-training given the ~2.24M parameter architecture; ADNI pre-training provides necessary weight initialization diversity.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.2.2 (software requirements include input data specifications), §5.7.5 (software integration testing — assay mismatch as integration risk)
- *ISO 14971 Reference:* Hazard H-012 (assay substitution introduces systematic bias in fluid modality output); Severity: Moderate; Probability: Medium (well-characterized in literature); Risk: Acceptable with transparency controls. Residual control: assay type captured in DICOM SR metadata at inference; mismatch flagged to operator.

---

### TDL-2B-003 — Synthesized Acoustic and Motor Features in ADNI Dataset

**Decision Made:**
Acoustic (speech) and motor (gait/tremor) features used in ADNI training are synthesized from published clinical distributions (reference: DRD-001). This is documented as a material limitation. The model is not validated for these modalities on real ADNI patient data. Bio-Hermes-001 fine-tuning uses real acoustic and motor acquisitions where available; modality-specific performance on real data is evaluated on Bio-Hermes-001 only.

**Rationale:**
ADNI does not collect standardized acoustic or motor digital biomarker data. These modalities are core to the NeuroFusion-AD multimodal architecture and represent a key differentiator for the intended use. Excluding them from pre-training would require a reduced architecture that would need to be expanded at fine-tune time, creating version discontinuity. Synthesized features drawn from clinical distributions preserve the input space dimensionality and allow gradient flow through the relevant encoder pathways, preventing weight starvation prior to real-data fine-tuning.

**Alternatives Considered:**
1. *Train a reduced architecture (fluid + clinical only) on ADNI, then add acoustic/motor at fine-tune:* Rejected. Architectural expansion post-pre-training is not supported by IEC 62304 §5.5 (software unit implementation) without a formal change request, and resets the development baseline.
2. *Source real acoustic/motor data from a third dataset for ADNI augmentation:* Not feasible within programme timeline and budget. Would require additional IRB and DUA review.
3. *Mask acoustic/motor loss terms during ADNI training:* Considered as a partial mitigation. Not adopted in favour of the synthesis approach, but noted as a valid alternative for future programme phases. The attention mechanism may assign spurious weight to synthesized modalities; the attention weights reported for ADNI (acoustic: 0.262; motor: 0.240) should be interpreted with caution and are flagged as unreliable in NFD-VVR-002.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.2.1 (software requirements shall reflect the intended use environment), §5.8.3 (software release — known anomalies must be documented)
- *ISO 14971 Reference:* Hazard H-015 (model outputs influenced by non-patient-derived training signals); Severity: Moderate; Probability: Low (synthesized features are replaced by real data at fine-tune; effect attenuated); Risk: Acceptable with residual disclosure in IFU and operator guidance. **Limitation note for IFU:** "Acoustic and motor modality contributions on ADNI-trained weights reflect synthetic distributions. Attention weights for these modalities derived from ADNI evaluation are not indicative of real-world modality utility."

---

### TDL-2B-004 — Architecture Reduction: 768→256 Embedding Dimension; ~60M→~2.24M Parameters

**Decision Made:**
The Phase 2B model architecture was reduced from the Phase 1/2A configuration (embed_dim=768, ~60M parameters) to embed_dim=256, dropout=0.4, ~2.24M parameters.

**Rationale:**
Three converging factors drove this decision:
1. *Dataset scale:* Post-remediation, the effective training set (ADNI: 345 + Bio-Hermes-001: 661 = 1,006 samples combined) is insufficient to reliably train a 60M parameter model. The Phase 2A configuration was at high risk of overfitting, particularly following leakage removal which reduces the effective signal-to-noise ratio.
2. *HPO signal:* Optuna HPO (15 trials) consistently favoured smaller embedding dimensions across search space explorations, with val AUC degrading in larger configurations under the constrained data regime.
3. *Deployment target:* The Navify Algorithm Suite inference environment specifies memory and latency budgets that the ~60M parameter model approached but did not comfortably meet on the target hardware profile. The 2.24M parameter model provides substantial margin.
The reduction in dropout from a potentially lower value to 0.4 was specifically targeted at the regularization deficit observed in Phase 2A training curves.

**Alternatives Considered:**
1. *Retain 768 embed_dim with aggressive regularization (dropout ≥ 0.6, weight decay):* Rejected. Regularization cannot compensate for fundamental parameter-to-data ratio mismatch at this scale; validation curves showed persistent overfitting signatures regardless of dropout value in HPO.
2. *Use a pretrained foundation model backbone (e.g., BioViL-T, clinical BERT for tabular):* Considered for future phases. Out of scope for Phase 2 given regulatory complexity of inherited pretraining data provenance.
3. *Intermediate reduction to embed_dim=512:* HPO trial data (W&B: t9s3ngbx) shows embed_dim=512 configurations achieved val AUC plateau ~0.01 below the 256-dim best trial; not selected.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.3.1 (software architectural design — design rationale documented), §5.3.6 (software architectural design review)
- *ISO 14971 Reference:* Risk reduction measure for H-016 (model overfitting → overconfident risk estimates in deployment); architecture reduction reduces overfitting probability. Residual risk: model may underfit rare presentations; addressed by Bio-Hermes subgroup analysis (see §4).

---

### TDL-2B-005 — HPO Budget Constraint: 15 Optuna Trials

**Decision Made:**
Hyperparameter optimization was conducted using Optuna with a budget of 15 trials on the ADNI dataset. The best trial achieved val AUC = 0.9081. No additional trials were performed.

**Rationale:**
The 15-trial budget reflects a hardware-constrained programme resource allocation (single RTX 3090). At the model scale of ~2.24M parameters, 15 trials provide reasonable coverage of the most impactful hyperparameter dimensions (embedding dimension, dropout rate, learning rate, weight decay, batch size). Optuna's TPE sampler is effective in low-budget regimes relative to grid or random search.

**Alternatives Considered:**
1. *Increase to 50+ trials:* Infeasible within timeline; estimated compute requirement 3× current budget on available hardware.
2. *Manual grid search over 3–4 key parameters:* Considered but rejected in favour of Optuna TPE, which is more sample-efficient and produces a searchable audit trail in W&B (t9s3ngbx).
3. *Use a cloud compute burst:* Not approved under current programme budget cycle. Flagged for Phase 3 planning.

**Limitation acknowledged:** 15 trials constitutes a constrained search. There is non-negligible probability that the globally optimal hyperparameter configuration within the defined search space was not identified. The best-trial val AUC (0.9081) is reported as a point estimate; the HPO process is not certified to have found a global optimum. This limitation is recorded in NFD-VVR-002 §2.1.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.1.1 (development planning — resource constraints documented), §5.4.1 (software unit implementation — implementation follows design)
- *ISO 14971 Reference:* Low-severity programmatic risk; no direct patient safety pathway. HPO suboptimality would manifest as modestly reduced AUC, which is captured by the validation confidence intervals.

---

### TDL-2B-006 — Bio-Hermes-001 Fine-Tuning Strategy: Frozen Encoders, Classification-Only Loss

**Decision Made:**
Bio-Hermes-001 fine-tuning froze all modality-specific encoder weights and updated only the classification head (final fusion layers and output projection). Loss function: binary cross-entropy on amyloid classification. No auxiliary regression losses (MMSE, survival) were active during fine-tuning.

**Rationale:**
Bio-Hermes-001 provides N=661 training samples. Unfreezing the full ~2.24M parameter network on this sample size risks catastrophic forgetting of the ADNI pre-trained representations and overfitting to the Bio-Hermes-001 class distribution. Freezing encoders preserves the multimodal feature representations learned on ADNI while allowing the classification head to adapt to: (a) the correct pTau217 assay (replacing pTau181 proxy), (b) the Bio-Hermes-001 demographic distribution (24% underrepresented communities), and (c) the Bio-Hermes-001 data collection environment.
The decision to exclude auxiliary losses (MMSE RMSE, survival C-index) during fine-tuning reflects the cross-sectional-only nature of Bio-Hermes-001; longitudinal outcomes are unavailable, making auxiliary loss supervision impossible without introducing imputed targets.

**Alternatives Considered:**
1. *Full fine-tuning (unfreeze all layers):* Rejected due to overfitting risk at N=661 and catastrophic forgetting risk as described above.
2. *Layer-wise learning rate decay (LLRD) fine-tuning:* A valid alternative; not adopted due to additional HPO complexity incompatible with the 15-trial budget constraint. Recommended for Phase 3.
3. *Include MMSE auxiliary loss with imputed Bio-Hermes-001 trajectories:* Rejected. Imputed longitudinal targets would introduce a secondary synthetic data source, compounding the limitations already present from ADNI acoustic/motor synthesis (TDL-2B-003).
4. *Train a separate Bio-Hermes-001-only model:* Rejected. Pre-training on ADNI provides critical sample diversity; a Bio-Hermes-001-only model would lack the ADNI age/APOE stratum representation necessary for the target population.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.5.3 (software unit verification — verify that fine-tuning produces verified outputs), §5.7.4 (software integration testing — verify that frozen encoder outputs remain stable under fine-tune)
- *ISO 14971 Reference:* Frozen encoder strategy is a risk control measure for H-018 (distribution shift between ADNI and deployment population causing miscalibrated outputs); the classification-head-only update confines adaptation to the output layer where calibration correction (temperature scaling) is also applied.

---

### TDL-2B-007 — Calibration Method: Temperature Scaling

**Decision Made:**
Post-hoc probability calibration was performed using temperature scaling on the ADNI validation set. Optimal temperature T = 0.76 was identified by minimising Expected Calibration Error (ECE). ECE reduced from 0.1120 (pre-calibration) to 0.0831 (post-calibration). Temperature parameter T = 0.76 is stored in the production model configuration and applied at inference time.

**Rationale:**
Neural network classifiers are known to produce overconfident probability estimates (Guo et al., 2017). The pre-calibration ECE of 0.1120 exceeds the programme-defined acceptable threshold of ECE ≤ 0.10 (NFD-SRS-002 §4.2.3). Temperature scaling is the simplest parametric calibration method, introducing a single scalar parameter with no risk of calibration set overfitting. For a Class IIa SaMD providing risk probability estimates to clinicians, calibration quality directly impacts the clinical utility and safety of the output; an overconfident model would systematically overstate amyloid progression risk.

T = 0.76 < 1.0 indicates the model was overconfident (logit magnitudes were too large); temperature scaling softens the probability outputs toward the empirical base rate.

**Alternatives Considered:**
1. *Platt scaling (logistic regression on logits):* Two-parameter method; provides marginally better calibration in some settings but risks overfitting on small calibration sets. Temperature scaling preferred for robustness.
2. *Isotonic regression:* Non-parametric; high overfitting risk on ADNI validation set (N=74). Rejected.
3. *No calibration:* ECE = 0.1120 exceeds threshold; not acceptable per NFD-SRS-002.
4. *Recalibrate on Bio-Hermes-001 validation set:* Considered but not implemented due to risk of domain-specific overfitting; ADNI calibration is more conservative and generalisable. Bio-Hermes-001 calibration performance to be evaluated in Phase 3.

**Limitation:** Post-calibration ECE = 0.0831 does not meet the ECE ≤ 0.10 threshold but falls short of the stretch goal of ECE ≤ 0.05. The residual miscalibration is disclosed in the IFU and is subject to ongoing monitoring via PMS drift detection (see §6).

**Risk Assessment:**
- *IEC 62304 Reference:* §5.7.3 (regression testing — calibration verified against pre-calibration baseline), §5.8.3 (known anomaly: residual ECE = 0.0831 documented at release)
- *ISO 14971 Reference:* Calibration is a risk control measure for H-019 (overconfident probability output causes clinician over-reliance); post-control residual risk: Low. IFU to include instruction: "Output probabilities reflect model confidence and are not equivalent to population-level incidence rates; clinical judgment is required."

---

### TDL-2B-008 — Classification Threshold Selection: 0.6443

**Decision Made:**
The operating classification threshold was set at 0.6443 (Youden's J-optimal on ADNI validation set), yielding Sensitivity = 0.793, Specificity = 0.933, PPV = 0.958, NPV = 0.700 on the ADNI test set.

**Rationale:**
The threshold was selected to maximise the Youden Index (Sensitivity + Specificity − 1) on the held-out ADNI validation set, representing a balanced operating point. In the MCI amyloid progression context, the asymmetry of clinical consequences favours high specificity to avoid unnecessary downstream invasive workup (e.g., CSF tap, PET imaging), while maintaining acceptable sensitivity. The observed specificity of 0.933 and PPV of 0.958 at this threshold are consistent with a tool intended to support referral decisions where false positives carry significant patient burden.

**Alternatives Considered:**
1. *Sensitivity-prioritised threshold (lower threshold):* Would increase sensitivity at the cost of specificity. In populations with lower amyloid prevalence than ADNI (which skews toward amyloid-positive MCI), the PPV reduction would be clinically significant.
2. *Fixed threshold of 0.50:* Suboptimal; model outputs are calibrated but not symmetric around 0.5 given class imbalance in ADNI.
3. *Clinician-adjustable threshold:* Architecturally supported by the Navify Algorithm Suite API. The production interface will expose threshold adjustability to site administrators within a constrained range [0.50, 0.75], enabling local calibration to site-specific population prevalence. This is documented as a configuration item in NFD-SDD-002.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.2.5 (software requirements — threshold as a configurable parameter), §5.8.7 (software release — threshold value stored in validated configuration file)
- *ISO 14971 Reference:* H-020 (threshold misapplication); residual control: threshold value displayed alongside output probability in the Navify SR report; administrator guidance provided in IFU §7.3.

---

### TDL-2B-009 — Stratified 70/15/15 Train/Val/Test Split for Bio-Hermes-001

**Decision Made:**
Bio-Hermes-001 (N=945) was partitioned using stratified random sampling: 70% train (N=661), 15% validation (N=142), 15% test (N=142). Stratification variables: amyloid status, age group (lt65/65–75/gt75), sex, APOE4 carrier status. Test set was held out through all training and HPO activity.

**Rationale:**
Stratified splitting ensures that the 24% underrepresented community participants are proportionally represented across all three partitions, preventing the validation or test set from being demographically unrepresentative. The 70/15/15 ratio is standard for datasets of this size class, balancing the need for adequate fine-tuning data against statistically meaningful evaluation set sizes (N=142 per split).

**Alternatives Considered:**
1. *80/10/10 split:* Increases training data but reduces test set to N=94, providing lower-precision AUC confidence intervals. Rejected.
2. *Use full Bio-Hermes-001 for evaluation only (no fine-tuning):* Would provide a cleaner external validation but sacrifices the assay-adaptation benefit of fine-tuning. Given the pTau181→pTau217 mismatch, fine-tuning is considered necessary.
3. *K-fold cross-validation:* Appropriate for model selection but does not provide a single held-out test estimate required for regulatory performance claims. Not adopted as primary evaluation strategy; may supplement in Phase 3.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.7.1 (software integration testing plan), §5.8.1 (software release testing)
- *ISO 14971 Reference:* Data split methodology is a risk control for H-021 (evaluation bias from non-representative test set); stratification reduces this risk to residual level. Residual: Bio-Hermes-001 is cross-sectional; longitudinal performance remains unvalidated (see PMS §6).

---

### TDL-2B-010 — Subgroup Fairness: APOE4 Carrier AUC Gap — No Model Change, Remediation Plan Required

**Decision Made:**
The subgroup fairness analysis identified a maximum AUC gap of 0.225 (APOE4 non-carrier: AUC = 0.906 vs. APOE4 carrier: AUC = 0.775), resulting in fairness_pass = False. A model change was not implemented at this stage. Instead, a formal fairness remediation plan is initiated as a Phase 3 pre-condition.

**Rationale:**
The AUC gap of 0.225 in APOE4 carriers is clinically meaningful and constitutes a performance disparity that must be addressed before broad deployment in APOE4-heterogeneous populations. However, the ADNI test set for the APOE4 carrier subgroup contains only N=36 subjects, and the 95% CI for the APOE4 carrier AUC is extremely wide (0.416–1.0), indicating that the point estimate of 0.775 is statistically unreliable. Implementing a model change in response to a noisy estimate carries risk of degrading overall performance without confirmed benefit. The appropriate response is: (1) flag this as a known limitation; (2) constrain deployment recommendations pending further validation; (3) mandate APOE4-stratified evaluation with adequate power (N≥100 per stratum) in Phase 3.

The age_lt65 subgroup AUC = 1.0 (N=11) is noted as almost certainly an artefact of small sample size and should not be interpreted as indicating superior performance in younger patients.

**Alternatives Considered:**
1. *Re-weight training data to oversample APOE4 carriers:* Insufficient APOE4 carrier representation in current datasets to implement meaningfully; would require new data acquisition.
2. *Train a separate APOE4-stratified model:* Out of scope for Phase 2; increases regulatory complexity. Recommended for Phase 3 evaluation.
3. *Block deployment in APOE4 carriers:* Operationally impractical and clinically inappropriate; APOE4 status is a key risk factor and the tool should ultimately be more, not less, informative for this subgroup. Targeted improvement is the correct path.

**Risk Assessment:**
- *IEC 62304 Reference:* §5.8.3 (known anomaly at release — documented in release notes), §5.1.9 (software risk management — unresolved risk items carried forward with control measures)
- *ISO 14971 Reference:* H-022 (differential performance by genetic subgroup — APOE4 carriers may receive less accurate risk estimates); Severity: Serious; Probability: Medium (gap confirmed in point estimate but CI wide); Risk: As Low As Reasonably Practicable (ALARP) pending Phase 3. Interim control: IFU restriction "Performance in APOE