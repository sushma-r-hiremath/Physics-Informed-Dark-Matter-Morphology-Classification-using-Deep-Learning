# Physics‑Informed Dark Matter Morphology Classification using Deep Learning

## 📌 Project Overview

This project explores **physics‑informed machine learning for strong gravitational lensing**, inspired by the **DeepLense (ML4Sci / GSoC)** initiative. The goal is to classify simulated strong lensing images into different **dark matter morphology classes** by combining:

* **Domain‑inspired simulation** of gravitational lensing
* **Convolutional Neural Networks (CNNs)**
* **Physics‑informed residual features** highlighting substructure

The project is implemented end‑to‑end in **Google Colab** and is fully reproducible.

---

## 🎯 Problem Statement

Strong gravitational lensing images encode information about the underlying **dark matter distribution** in lensing galaxies. Subtle perturbations in Einstein rings can arise from:

* Smooth dark matter halos (NO_SUB)
* Cold Dark Matter (CDM) subhalos
* Axion‑like or wave‑like dark matter effects

The challenge is that **pixel‑level intensity distributions strongly overlap**, making classification non‑trivial. This motivates the use of **spatial feature learning** and **physics‑guided representations**.

---

## 🧪 Dataset Generation

### Why a Custom Simulator?

The official DeepLense datasets are not directly downloadable, and astrophysical simulators (e.g. `lenstronomy`) are currently **incompatible with Google Colab’s Python/Numba environment**. To ensure stability and reproducibility, a **custom analytical lensing simulator** was implemented.

### Simulation Details

Each image is generated using:

* A **background Sérsic galaxy** (source)
* A **singular isothermal sphere (SIS)** lens potential
* Ray‑tracing via a simplified lens equation
* Gaussian PSF and observational noise

### Classes

| Class  | Description                                             |
| ------ | ------------------------------------------------------- |
| NO_SUB | Smooth lens with no substructure                        |
| CDM    | Random small‑scale perturbations mimicking CDM subhalos |
| AXION  | Coherent wave‑like perturbations (axion‑inspired proxy) |

> ⚠️ *Note:* The axion class is a **proxy**, not a full physical axion field simulation.

### Dataset Summary

* Image size: **64 × 64**
* Channels: 1 (baseline) or 2 (physics‑informed)
* Samples per class: **500**
* Total samples: **1500**

---

## 🧠 Machine Learning Approach

### 1️⃣ Baseline CNN

A standard CNN is trained on **raw lensing images only**.

Architecture:

* 3 convolutional blocks (Conv → ReLU → MaxPool)
* Fully connected classifier
* Cross‑entropy loss

Purpose:

* Establish a reference performance
* Evaluate how much information is already captured by morphology

---

### 2️⃣ Physics‑Informed CNN (Key Contribution)

To inject domain knowledge, we introduce a **residual channel**:

```
Residual = Original Image − Smoothed Image
```

This highlights:

* Small‑scale perturbations
* Deviations from smooth lensing
* Substructure signatures

The CNN now receives a **2‑channel input**:

1. Original image
2. Residual (physics‑informed feature)

Additional improvements:

* Batch Normalization
* Stronger Dropout
* Data augmentation (horizontal/vertical flips)

---

## 📊 Results

### Training Dynamics

* Both models converge stably
* Physics‑informed model converges **faster**
* Validation accuracy consistently higher for physics‑informed CNN

### Confusion Matrix Analysis

* Baseline CNN shows confusion between **CDM ↔ AXION**
* Physics‑informed CNN significantly reduces this confusion

### Key Insight

> Even when pixel‑level statistics overlap, **embedding physical structure into the learning pipeline improves class separability**.

---

## 🔍 Visualization Notes

Strong lensing images have **high dynamic range**:

* Most pixels ≈ 0
* Signal localized along thin Einstein rings

Therefore:

* Linear visualization appears nearly black
* Logarithmic and percentile‑based scaling are required

This follows **standard astronomical imaging practice**.

---

## 🧱 Project Structure

```
Physics-Informed-Dark-Matter-Lensing/
│
├── data/
│   └── raw/
│       ├── NO_SUB/
│       ├── CDM/
│       └── AXION/
│
├── notebooks/
│   ├── 01_simulation.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_physics_informed_cnn.ipynb
│   └── 05_results_analysis.ipynb
│
├── src/
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix_baseline.png
│   └── confusion_matrix_physics.png
│
├── README.md
└── requirements.txt
```

---

## 🧩 Issues Faced & How They Were Solved

### 1️⃣ Library Incompatibility (lenstronomy + Colab)

* **Issue:** `numba.generated_jit` errors due to Python 3.12
* **Solution:** Replaced dependency with a custom analytical simulator

### 2️⃣ Black / Empty Visualizations

* **Issue:** Images appeared black despite valid data
* **Cause:** Extremely low mean intensity & high dynamic range
* **Solution:** Log‑scale and percentile‑based visualization

### 3️⃣ Google Drive Showing Empty Folders

* **Issue:** Drive UI inconsistent with Python filesystem
* **Solution:** Verified data using `os.listdir`; trusted Python paths

### 4️⃣ Data Augmentation Errors

* **Issue:** Augmentation code executed outside Dataset scope
* **Solution:** Moved augmentation into `Dataset.__getitem__`

These challenges reflect **real‑world ML engineering and research debugging**.

---

## 🚀 Future Work

* Replace proxy axion perturbations with field‑theoretic simulations
* Incorporate observational effects (PSF variation, noise models)
* Extend to regression tasks (subhalo mass estimation)
* Apply explainability methods (Grad‑CAM on residual channel)

---

## 🏁 Conclusion

This project demonstrates that **physics‑informed representations significantly enhance machine learning performance** in scientific imaging tasks. By combining domain knowledge with deep learning, we achieve better generalization and interpretability — aligning closely with the goals of **AI for Science**.

---

## 🙏 Acknowledgements

Inspired by the **DeepLense / ML4Sci / Google Summer of Code** projects and the broader AI‑for‑Science community.

