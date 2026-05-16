# Cr3+SpectraPredictor

## Predicting Emission Wavelength and FWHM of Cr³⁺-Substituted Phosphors
This repository provides machine learning models for predicting the emission peak wavelength (λem) and full width at half maximum (FWHM) of Cr3+-substituted phosphors using compositional and structural descriptors.

By combining the predicted λem and FWHM values with a Gaussian function, an estimated emission spectrum can be generated for Cr3+-substituted phosphor.

---

## 📑 Table of Contents

- 📚 Citations
- ⚙️ Prerequisites
- 🚀 Usage
- 📄 Define Composition Features
- 🏗️ Define Structural Features
- 📄 Define the Prediction Set
- 🔮 Predict λem and FWHM
- 👨‍💻 Authors

---

## 📚 Citations

To cite the Cr³⁺ emission and FWHM prediction models, please reference the following work:

Amit Kumar, et al., “Machine Learning-Assisted Discovery of Cr³⁺-Based NIR Phosphors” (Submitted).

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
