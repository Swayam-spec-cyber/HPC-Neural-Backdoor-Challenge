# SPECTRA-CAD: Hardware Performance Counter Neural Backdoor Detector

**Problem Title:** Detection of Backdoor Attacks using Hardware Performance Counters (HPCs)  
**Framework:** SPECTRA-CAD (**S**pectral-**C**opula & **P**arametric **S**tructural **I**nvariant **C**onformal **A**nomaly **D**etector)  
**Target Domain:** Black-Box Neural Network Security via Microarchitectural Side-Channels

---

## 1. Overview & Problem Statement

In a neural backdoor (trojan) attack, an untrusted party trains a deep neural network such that it behaves normally on standard benign inputs, but selectively alters its prediction whenever a hidden trigger pattern is embedded in the input. Under a strict black-box hardware defense scenario, internal model weights, gradients, and architectures are inaccessible. Instead, the defender monitors low-level **Hardware Performance Counters (HPCs)** measured from the processor during model inference.

Given $N = 800$ clean (benign) inference traces in `trace.csv` across $d = 3$ hardware counters (`cache-references`, `cycles`, and `LLC-loads`) and **zero backdoor training examples**, **SPECTRA-CAD** learns the physical conservation and coupling laws of clean model execution to flag backdoor-triggered inferences in real time.

---

## 2. Key Technical Contributions

Standard anomaly detectors used in prior literature (such as Gaussian Mixture Models, Isolation Forest, and One-Class SVM) treat HPC columns as independent or globally elliptical tabular features. Consequently, they suffer from threshold inflation caused by operating-system (OS) interrupt spikes and fail when adaptive backdoor triggers break inter-counter physical coupling without exceeding marginal feature bounds.

**SPECTRA-CAD** introduces four core innovations tailored to the microarchitectural manifold of `trace.csv`:

1. **Closed-Form Heteroscedastic Structural Equations (CC-SEM):** Models each hardware counter $x_j$ as a degree-3 polynomial function of the remaining counters $x_{-j}$ via an ensemble of $\ell_2$-regularized Ridge and Huber regressors, paired with a degree-2 parametric log-variance estimator that normalizes local conditional heteroscedasticity. In addition, a dedicated 1-to-1 conditional propagation law polices the strong rank coupling ($\rho = 0.8962$) between `cache-references` and `LLC-loads`.
2. **Parametric 64-Knot Order-Statistic Gaussian Copula & Aitchison Log-Simplex Geometry:** Projects skewed hardware counts into a non-Gaussian rank copula space using $Q = 64$ quantile knots and computes scale-invariant log-ratios that mathematically cancel multiplicative clock/OS load jitter:
   $$\text{LR}_{jk}(x) = \ln\left(\frac{x_j + 1}{x_k + 1}\right)$$
3. **Orthogonal Minor-Subspace Spectral Projection (SPE):** Projects scaled traces onto the smallest eigenvectors ($\lambda_3 \to 0$) of the decontaminated covariance matrix to isolate low-amplitude adaptive triggers hiding in the rigid invariant subspace.
4. **OS-Decontaminated Out-of-Fold Conformal Calibration & $L_p$ ($p = 5$) Fusion:** Automatically filters out the $\sim 2.5\%$ benign OS-interrupt spikes (`cycles` reaching $4.996 \times 10^9$, $+8.46\sigma$ above the mean) before calibrating Out-of-Fold Empirical Cumulative Distribution Function (ECDF) survival scores. Pillar scores are combined via a generalized $L_5$ power mean so localized invariant violations are never diluted by quiet marginal views.

---

## 3. Time & Space Complexity Guarantees

Unlike memory-based anomaly detectors (`LOF`, $k$-NN, or dense kernel `OneClassSVM`) that store all $N$ training samples and require $\mathcal{O}(N \cdot d)$ complexity per test inference, **SPECTRA-CAD** is strictly parametric during matrix evaluation:

| Metric | Asymptotic Complexity | Empirical Measurement on `trace.csv` ($N = 800, d = 3, D = 12$) |
| :--- | :--- | :--- |
| **Offline Training Time** | $\mathcal{O}(N \cdot D^2 + N \log N \cdot d)$ | $\sim 0.45\text{ s}$ on standard CPU (no GPU required) |
| **Online Inference Latency** | $\mathcal{O}(K \cdot D^2 + \log N)$ per trace | $\sim 0.03\text{ ms}$ per trace ($\mathcal{O}(1)$ w.r.t. $N$ for matrix ops) |
| **Saved Model Footprint** | $\mathcal{O}(K \cdot D^2 + Q \cdot d)$ | $\sim 42\text{ KB}$ serialized `.joblib` file (stores zero raw training rows) |

*(Here $d = 3$ raw counters, $D = 12$ total engineered invariants, $K = 3$ mixture modes, and $Q = 64$ copula knots).*

---

## 4. Repository Structure

```text
final_hackathon_submission/
├── data/
│   └── trace.csv                        # 800 benign HPC inference traces (cache-references, cycles, LLC-loads)
├── src/
│   ├── __init__.py                      # Package initializer
│   └── detector.py                      # Core classes: ParametricCopulaTransform, MicroarchitecturalFeatureEngine, SpectraCADDetector
├── saved_models/
│   └── hpc_detector.joblib              # Final pre-trained SPECTRA-CAD model fit on all 800 benign traces
├── figures/
│   └── report_diagnostic_plot.png       # High-resolution score separation and F1 robustness plots
├── train.py                             # Standalone CLI script to train and serialize the detector
├── predict.py                           # Standalone CLI script for validation/test inference & root-cause attribution
├── ablation_benchmark_results.csv       # Full benchmark evaluation table across 4 backdoor attack families
├── report.pdf                           # 2-page IEEE-style technical report describing the method and results
├── requirements.txt                     # Python package dependencies
└── README.md                            # Repository documentation
```

---

## 5. Installation & Quickstart Usage

### Prerequisites
Install the lightweight CPU dependencies (Python $3.9+$):

```bash
pip install -r requirements.txt
```

### A. Running Inference on Unseen Validation / Test Traces (`predict.py`)
To evaluate the saved detector on any CSV file containing the HPC columns (`cache-references`, `cycles`, `LLC-loads`):

```bash
python predict.py \
    --input data/trace.csv \
    --model saved_models/hpc_detector.joblib \
    --output predictions.csv
```

#### Output Format (`predictions.csv`)
The inference script generates a CSV with four columns designed to support both threshold-independent metrics (**AUROC**) and threshold-dependent metrics (**Accuracy, TPR, FPR, F1**), plus physical explainability:
* `anomaly_score`: Continuous calibrated anomaly score in $[0, \infty)$ for computing **AUROC**.
* `conformal_pvalue`: Distribution-free inductive conformal $p$-value in $(0, 1]$ indicating the empirical tail probability under the clean hardware manifold.
* `prediction`: Binary classification label (`0` = Clean / Benign inference, `1` = Backdoor / Trojan inference) computed via `anomaly_score >= detector.threshold_`.
* `dominant_invariant_violation`: Identifies which of the 6 microarchitectural pillars triggered the highest anomaly response (e.g., `P4_Pairwise_Cache_LLC_Propagation_Law` or `P5_3D_Orthogonal_Minor_Subspace_SPE`).

### B. Retraining the Detector from Scratch (`train.py`)
To retrain the detector with finalized parameters on a clean HPC dataset:

```bash
python train.py \
    --input data/trace.csv \
    --output saved_models/hpc_detector.joblib \
    --alpha 0.042
```

### C. Python API Usage (For Custom Automated Grading Scripts)
Evaluators can also load the serialized model directly in Python:

```python
import joblib
import pandas as pd

# Load saved detector
detector = joblib.load("saved_models/hpc_detector.joblib")

# Load test traces
df_test = pd.read_csv("data/trace.csv")[["cache-references", "cycles", "LLC-loads"]]

# Continuous anomaly scores (for AUROC)
scores = detector.score_samples(df_test)

# Binary predictions: 0 = Clean, 1 = Backdoor (for Accuracy, TPR, FPR, F1)
labels = detector.predict(df_test)

# Full diagnostic DataFrame with conformal p-values and root-cause pillar attribution
report_df = detector.explain_anomalies(df_test)
```

---

## 6. Forensic Dataset Profile & Experimental Benchmark Summary

### Forensic Summary of `trace.csv` ($N = 800$ Benign Inferences)

| HPC Counter | Min | 25% | Median (50%) | 75% | Max | Mean | Std Dev |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `cache-references` | $1.679 \times 10^8$ | $1.787 \times 10^8$ | $1.839 \times 10^8$ | $1.862 \times 10^8$ | $1.916 \times 10^8$ | $1.824 \times 10^8$ | $4.999 \times 10^6$ |
| `cycles` | $4.699 \times 10^9$ | $4.756 \times 10^9$ | $4.773 \times 10^9$ | $4.788 \times 10^9$ | $4.996 \times 10^9$ | $4.772 \times 10^9$ | $2.647 \times 10^7$ |
| `LLC-loads` | $5.916 \times 10^6$ | $6.403 \times 10^6$ | $6.657 \times 10^6$ | $6.777 \times 10^6$ | $7.108 \times 10^6$ | $6.593 \times 10^6$ | $2.475 \times 10^5$ |

* **Memory Hierarchy Coupling:** `cache-references` and `LLC-loads` share a Spearman rank correlation of $\rho = 0.8962$, with an invariant Last-Level Cache miss propagation ratio ($\text{LCRR} = \frac{\text{LLC-loads}}{\text{cache-references}}$) centered at $3.619\%$.
* **OS Interrupt Spikes:** While `cycles` has a narrow coefficient of variation ($\text{CV} = 0.55\%$), the upper tail reaches $4.996 \times 10^9$ ($+8.46\sigma$ above the mean), confirming the presence of benign OS context-switch/interrupt outliers that must be decontaminated prior to threshold calibration.

### Benchmark Comparison Across 4 Microarchitectural Backdoor Regimes
Evaluated on an $80/20$ split ($640$ clean training traces, $160$ unseen clean test traces + $160$ backdoor traces per attack scenario):

| Attack Scenario | Method | AUROC | F1 Score | TPR | FPR | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Attack A: Patch Trigger (Sparsity / Cache Shift)** | 2024 Baseline: Standard GMM | $1.0000$ | $0.9756$ | $1.0000$ | $0.0500$ | $0.9750$ |
| | 2024 Baseline: Isolation Forest | $0.9978$ | $0.9783$ | $0.9875$ | $0.0312$ | $0.9781$ |
| | 2024 Baseline: One-Class SVM | $1.0000$ | $0.9846$ | $1.0000$ | $0.0312$ | $0.9844$ |
| | **Proposed: SPECTRA-CAD (Ours)** | **$1.0000$** | **$0.9907$** | **$1.0000$** | **$0.0187$** | **$0.9906$** |
| **Attack B: Stealthy Blended Trigger (Low-Amp)** | 2024 Baseline: Standard GMM | $0.9956$ | $0.9598$ | $0.9687$ | $0.0500$ | $0.9594$ |
| | 2024 Baseline: Isolation Forest | $0.9658$ | $0.7500$ | $0.6187$ | $0.0312$ | $0.7937$ |
| | 2024 Baseline: One-Class SVM | $0.9878$ | $0.9423$ | $0.9187$ | $0.0312$ | $0.9437$ |
| | **Proposed: SPECTRA-CAD (Ours)** | **$0.9980$** | **$0.9812$** | **$0.9812$** | **$0.0187$** | **$0.9812$** |
| **Attack C: Pure Coupling Break (Same Marginals)** | 2024 Baseline: Standard GMM | $0.7526$ | $0.5882$ | $0.4375$ | $0.0500$ | $0.6937$ |
| | 2024 Baseline: Isolation Forest | $0.6787$ | $0.2162$ | $0.1250$ | $0.0312$ | $0.5469$ |
| | 2024 Baseline: One-Class SVM | $0.7631$ | $0.5897$ | $0.4312$ | $0.0312$ | $0.7000$ |
| | **Proposed: SPECTRA-CAD (Ours)** | **$0.9140$** | **$0.8425$** | **$0.7562$** | **$0.0250$** | **$0.8656$** |
| **Attack D: Adaptive Minor-Subspace Trigger** | 2024 Baseline: Standard GMM | $0.9236$ | $0.7126$ | $0.5812$ | $0.0500$ | $0.7656$ |
| | 2024 Baseline: Isolation Forest | $0.8117$ | $0.2723$ | $0.1625$ | $0.0312$ | $0.5656$ |
| | 2024 Baseline: One-Class SVM | $0.8428$ | $0.7059$ | $0.5625$ | $0.0312$ | $0.7656$ |
| | **Proposed: SPECTRA-CAD (Ours)** | **$0.9585$** | **$0.8968$** | **$0.8375$** | **$0.0250$** | **$0.9062$** |