# Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy: A novel pathway for musculoskeletal ultrasound analysis

This repository implements an automatic and reproducible pipeline for muscle ultrasound analysis intrducing a machine learning approach that integrates deep learning-based segmentation and classification with radiomics-driven Heckmatt grading. The goal is to enhance the objectivity and efficiency of muscle ultrasound evaluation, reducing reliance on time-consuming manual assessments and overcoming interobserver variability.

The automated process consists of:
- **Muscle Segmentation & Classification:** Using a multi-class K-Net architecture, the pipeline accurately differentiates and segments 16 muscle groups, achieving high Intersection over Union (IoU) scores across folds.
- **Quantitative Heckmatt Grading:** Radiomics features are extracted from segmented muscles and their deeper regions, with an XGBoost classifier assigning a modified Heckmatt score (Normal, Uncertain, Abnormal). SHAP analysis further provides interpretability by pinpointing critical features driving the scoring decisions.

The results demonstrate high segmentation accuracy and robust grading performance, with an Area Under Curve (AUC) of up to 0.97 for abnormal cases. Automating these tasks improves diagnostic consistency and supports clinical decision-making in neuromuscular disease evaluation, particularly for conditions such as facioscapulohumeral muscular dystrophy (FSHD).

The repository is structured as follows:
- **`mmsegmentation/`**: Contains deep learning code for muscle segmentation & classification with K-Net.
- **`feature_extraction/`**: Scripts to extract texture/radiomics features and evaluate segmentation metrics.
- **`prediction_heckmatt/`**: Implements Heckmatt scoring with XGBoost on the extracted radiomics features.


### 1. **`mmsegmentation/`**
Contains **deep learning** code for muscle segmentation & classification with **K-Net** (based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)):

- **`tools/train.py`**  
  Train your segmentation model (multi-class or muscle-specific).
- **`tools/test.py`**  
  Evaluate a trained model on a test set.
- **`tools/local_inference.py`**  
  Quick local inference for debugging.
- **`utils/compareRevisionResults.py`**  
  Compare segmentation metrics (IoU, precision, recall) or different model revisions with statistical tests.

### 2. **`feature_extraction/`**
Scripts to **extract texture/radiomics features** and evaluate segmentation metrics:

- **`extractNormalizedTextureFeaturesFast.py`**  
  Extracts radiomics features with PyRadiomics, writing to JSON/Excel.  
- **`computeMetricsAndValuesFast.py`**  
  Generates confusion matrices, classification reports, IoU stats, etc.  
- **`readDicom.py`** & **`readSAV.py`**  
  Helpers for reading DICOM or `.sav` data, not strictly needed if you already have PNG images/tabular data.

### 3. **`prediction_heckmatt/`**
Implements **Heckmatt scoring** with **XGBoost** on the extracted radiomics features:

- **`PredictHeckmattScore_XG_plus_shap.py`**  
  Uses both CSA & deeper region features to classify 3 classes (Normal/Uncertain/Abnormal). Includes SHAP interpretability.
- **`PredictHeckmattScore_XG_plus_shap_onlyCSA.py`**  
  Ablation using **only** CSA features.
- **`PredictHeckmattScore_XG_plus_shap_onlyNOT.py`**  
  Ablation using **only** the deeper region.
---

## How to Reproduce the Results

1. **Install Dependencies & Environment**
   - Python ≥ 3.8 recommended.
   - Key packages: PyTorch, MMCV, MMEngine, MMSegmentation, PyRadiomics, XGBoost, SHAP, pandas, seaborn, etc.
   - The `mmsegmentation` folder is a partial copy of [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). Ensure versions match `mmseg/__init__.py`.

2. **Prepare the Ultrasound Dataset**
   - Data used in the paper: [Mendeley dataset](https://doi.org/10.17632/yzg86vb895.1).
   - Split images into train/test folds or replicate the 5-/10-fold cross-validation as in the paper.

3. **Train the Segmentation Model**
   - Under `mmsegmentation/tools/`, adapt or create a config for K-Net (similar to `knet_swin_mod`).
   - Example:
     ```bash
     python train.py /path/to/your_config.py --work-dir /path/to/save/checkpoints
     ```
   - This trains a multi-class muscle segmentation. For each muscle, fine-tune if needed.

4. **Evaluate & Generate Segmentation Maps**
   - Use `test.py`:
     ```bash
     python test.py /path/to/your_config.py /path/to/checkpoint.pth
     ```
   - Saves predictions (PNG). Then run `computeMetricsAndValuesFast.py` or for confusion matrices, IoU, etc.

5. **Extract Radiomics Features**
   - In `feature_extraction/`, edit **`extractNormalizedTextureFeaturesFast.py`** to point to your predicted masks & raw images.
   - Run:
     ```bash
     python extractNormalizedTextureFeaturesFast.py
     ```
   - Outputs a JSON/Excel summary with features. Make sure it includes "manual_h_score" if you want to replicate the classification steps.

6. **Heckmatt Classification**
   - Go to `prediction_heckmatt/`, pick a script:
     - `PredictHeckmattScore_XG_plus_shap.py`: combined CSA+deeper region features.
     - `PredictHeckmattScore_XG_plus_shap_onlyCSA.py`: ablation using CSA only.
     - `PredictHeckmattScore_XG_plus_shap_onlyNOT.py`: ablation using deeper region only.
   - Verify it reads the features from the prior step.  
   - It runs 10-fold CV with XGBoost, producing confusion matrices, ROC curves, and SHAP:
     ```bash
     python PredictHeckmattScore_XG_plus_shap.py
     ```
   - The resulting classification metrics match the paper's results.

---

## Citation & License

If you find this pipeline useful, please cite our paper:

```bibtex
@article{MARZOLA2025,
title = {Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy: A novel pathway for musculoskeletal ultrasound analysis},
journal = {Clinical Neurophysiology},
year = {2025},
issn = {1388-2457},
doi = {https://doi.org/10.1016/j.clinph.2025.01.016},
url = {https://www.sciencedirect.com/science/article/pii/S1388245725000367},
author = {Francesco Marzola and Nens {van Alfen} and Jonne Doorduin and Kristen Mariko Meiburger},
keywords = {Muscle ultrasound, Machine learning, Muscle segmentation, Heckmatt grading, Neuromuscular disease diagnosis}}
