# Model Card — PneumoScan (Chest X-ray Pneumonia Triage)

**Version:** 2.0 · **Task:** Binary classification (pneumonia vs. not) on frontal chest radiographs, framed as **triage** (prioritization), not diagnosis.

> ⚠️ **Not a medical device.** This model is a research and educational prototype. It must not be used to make, support, or defer any real clinical decision.

---

## Intended use

- **Intended:** demonstrating a trustworthiness-first ML workflow — thresholding for sensitivity, external validation, calibration, uncertainty-aware abstention, subgroup fairness auditing, and explainability — on a well-known open dataset.
- **Intended users:** ML practitioners, reviewers, and students evaluating the *methodology*.

### Out of scope (do not use for)

- Any clinical diagnosis, screening, or triage of real patients.
- Populations, equipment, or views outside the training distribution (see Limitations).
- Non-frontal views, non-chest images, or modalities other than plain radiographs.
- Any setting where a wrong output could affect a person's care.

---

## Model details

- **Architecture:** DenseNet121 (ImageNet-pretrained) backbone + custom head
  `GlobalAveragePooling → Dense(512, ReLU) → BatchNorm → Dropout(0.3) → Dense(256, ReLU) → BatchNorm → Dropout(0.3) → Dense(1, sigmoid)`.
- **Training:** two-stage transfer learning (frozen-backbone head training, then top-conv-block fine-tuning). The **frozen-backbone stage produced the best validation AUC**; fine-tuning did not improve on it. Binary cross-entropy with class weights (~1.85 / 0.69), EarlyStopping and ReduceLROnPlateau on validation ROC-AUC.
- **Preprocessing:** resize to 224×224, scale to [0,1], ImageNet mean/std normalization ("torch" mode, matching DenseNet).
- **Decision pipeline:** raw sigmoid → temperature calibration (T = 0.72) → operating threshold (calibrated 0.265) → triage zone.
- **Serving:** DenseNet121 backbone exported to ONNX; head + Grad-CAM reimplemented in NumPy. Validated numerically identical to the TensorFlow original (prediction |Δ| ~1e-8; Grad-CAM correlation 1.0).

## Training data

- **Source:** Kaggle "Chest X-Ray Pneumonia" (Paul Mooney) — **pediatric** patients (Guangzhou Women and Children's Medical Center).
- **Re-split:** the original 16-image validation set was discarded; data was re-split 80/10/10 stratified by class (~73% pneumonia in each split).
- **Known bias:** pediatric-only, single-source. This is the root cause of the external-validation behavior below.

## Evaluation

- **Internal test** (held-out split, same distribution): sensitivity 0.942, specificity 0.962, precision 0.985, accuracy 0.947, ROC-AUC 0.986, PR-AUC 0.995.
- **External validation** — RSNA Pneumonia Detection Challenge (**adult**), n = 3,000, evaluated at the **frozen** internal threshold: sensitivity 0.907, specificity 0.467, ROC-AUC 0.782, precision 0.331.
- **Calibration:** temperature scaling reduced test ECE 0.080 → 0.072.
- **Triage:** with an abstention band, the model decides 93% of cases at 0.971 accuracy and escalates the rest.

Primary metric is **sensitivity on the pneumonia class** — accuracy is misleading under ~3:1 class imbalance.

---

## Limitations & risks

1. **Specificity collapses off-distribution.** On adult X-rays the model false-alarms heavily (specificity 0.96 → 0.47). The collapse is concentrated in the *"No Lung Opacity / Not Normal"* category (specificity 0.278) vs. true *Normal* (0.703) — the model over-flags abnormal-but-not-pneumonia findings it never saw in pediatric training.

2. **PA-view weakness.** Sensitivity is 0.985 on AP views but **0.615 on PA views** (n=143) — below an 0.85 acceptability floor on a non-trivial sample. The model should not be relied on for PA-view images.

3. **Explanations can be spurious.** Grad-CAM shows correct lung-field attention on many true positives, but several **false positives are driven by attention to image borders, the diaphragm, or acquisition markers** rather than pathology. High-confidence outputs are not guaranteed to be for the right reasons.

4. **Pediatric training source.** Generalization to adults, different scanners, or different acquisition protocols is not established and, where measured, is degraded.

5. **Sex/age subgroups** are stable in this audit (~0.88–0.93), but the audit is limited to the RSNA metadata available and is not a substitute for prospective validation.

## Ethical considerations

- Automation bias is a real risk: a plausible label plus a plausible-looking heatmap can over-persuade. The abstention/triage design and the prominent in-product disclaimer are partial mitigations, not solutions.
- No patient-identifiable data is stored by the demo; uploads are processed in-memory for a single prediction.
- This model has **not** been reviewed or cleared by any regulatory body (e.g., FDA/CE) and carries no such claim.

## How to cite / reproduce

Code, evaluation scripts, and figures: https://github.com/decstar0610/pneumonia-detection-cnn
Model artifacts: https://huggingface.co/decstzz06/pneumoscan-model
