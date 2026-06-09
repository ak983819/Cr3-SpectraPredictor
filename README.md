# Cr3+SpectraPredictor

![Graphical Overview](TOC-01.jpg)

## Predicting Emission Wavelength and FWHM of Cr³⁺-Substituted Phosphors

Cr3+SpectraPredictor is a machine-learning framework for predicting the emission peak wavelength (λem) and full width at half maximum (FWHM) of Cr³⁺-substituted phosphors using compositional and structural descriptors.

By combining the predicted λem and FWHM values with a Gaussian function, an estimated emission spectrum can be generated for a target Cr³⁺ phosphor.

---

## 📑 Table of Contents

* [Citation](#-citation)
* [Prerequisites](#️-prerequisites)
* [Usage](#-usage)

  * [1. Define Features for the Emission Model](#1-define-features-for-the-emission-model)
  * [2. Define Features for the FWHM Model](#2-define-features-for-the-fwhm-model)
  * [3. Predict Emission Wavelength (λem)](#3-predict-emission-wavelength-λem)
  * [4. Predict FWHM](#4-predict-fwhm)
  * [5. Generate a Gaussian Emission Spectrum](#5-generate-a-gaussian-emission-spectrum)
* [Authors](#-authors)

---

## 📚 Citation

If you use this repository, please cite:

**Amit Kumar et al.**
*Machine Learning-Assisted Discovery of Cr³⁺-Based NIR Phosphors* (Submitted).

---

## ⚙️ Prerequisites

This package requires the following Python libraries:

```text
pymatgen
catboost
scikit-learn
pandas
numpy
matplotlib
openpyxl
shap
```

Install them using:

```bash
pip install pymatgen catboost scikit-learn pandas numpy matplotlib openpyxl shap
```

---

## 🚀 Usage

### 1. Define Features for the Emission Model

Run:

```text
em_features.ipynb
```

Before running the notebook, ensure that the following files are located in the same directory:

* `Feature_generator.xlsx`
* `elements.xlsx`
* CIF files (`*.cif`)

The notebook generates:

```text
To_predict_em.xlsx
```

### Additional Features Required

This notebook generates 17 of the 22 features required by the emission model. To complete `To_predict_em.xlsx`, the following descriptors must be added manually:

* `max_metal_ligand_bond_length`
* `mean_metal_ligand_bond_length`
* `polyhedron volume`
* `distortion_index`
* `x` (Cr³⁺ dopant concentration)

The `1/R²` descriptor should be calculated using the ionic radii listed in the table provided below.

---

### 2. Define Features for the FWHM Model

Run:

```text
fwhm_features.ipynb
```

Before running the notebook, ensure that the following files are located in the same directory:

* `Feature_generator.xlsx`
* `elements.xlsx`
* CIF files (`*.cif`)

The notebook generates:

```text
To_predict_fwhm.xlsx
```

### Additional Features Required

This notebook generates 13 of the 18 features required by the FWHM model. To complete `To_predict_fwhm.xlsx`, the following descriptors must be added manually:

* `max_metal_ligand_bond_length`
* `mean_metal_ligand_bond_length`
* `polyhedron volume`
* `x` (Cr³⁺ dopant concentration)

The `R5` descriptor should be calculated using the ionic radii listed in the table provided below.

---

### 3. Predict Emission Wavelength (λem)

Run:

```bash
python em_model.py
```

Output:

```text
predicted_em.xlsx
```

---

### 4. Predict FWHM

Run:

```bash
python fwhm_model.py
```

Output:

```text
predicted_fwhm.xlsx
```

---

### 5. Generate a Gaussian Emission Spectrum

After obtaining the predicted λem and FWHM values, an approximate emission spectrum can be generated using a Gaussian profile:

[
I(\lambda)=I_0 \exp\left[-4\ln(2)\left(\frac{\lambda-\lambda_{em}}{\mathrm{FWHM}}\right)^2\right]
]

where:

* λ = wavelength
* λem = predicted emission peak wavelength
* FWHM = predicted full width at half maximum
* I₀ = peak intensity

---

## 👨‍💻 Authors

**Amit Kumar**
Department of Chemistry
University of Houston

**Jakoah Brgoch**
Department of Chemistry
University of Houston
