# Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy: A novel pathway for musculoskeletal ultrasound analysis

This repository implements an automatic and reproducible pipeline for muscle ultrasound analysis introducing a machine learning approach that integrates deep learning-based segmentation and classification with radiomics-driven Heckmatt grading. The goal is to enhance the objectivity and efficiency of muscle ultrasound evaluation, reducing reliance on time-consuming manual assessments and overcoming interobserver variability.

Heckmatt grading is usually performed by visual inspection by experienced clinicians. An example of Heckmatt grading is shown below:

![Figure1](Figure1.png)
*Examples of Heckmatt grading in the Rectus Femoris muscle. (A) Heckmatt score 1 (Normal): Normal muscle; (B) Heckmatt score 2 (Uncertain): Increased muscle echo intensity with distinct bone echo; (C) Heckmatt score 3 (Abnormal): Marked increased muscle echo intensity with a reduced bone echo; (D) Heckmatt score 4 (Abnormal): Very strong muscle echo and complete loss of bone echo.*

The automated process consists of:
- **Muscle Segmentation & Classification:** Using a multi-class K-Net architecture, the pipeline accurately differentiates and segments 16 muscle groups, achieving high Intersection over Union (IoU) scores across folds.
- **Quantitative Heckmatt Grading:** Radiomics features are extracted from segmented muscles and their deeper regions, with an XGBoost classifier assigning a modified Heckmatt score (Normal, Uncertain, Abnormal). SHAP analysis further provides interpretability by pinpointing critical features driving the scoring decisions.

![Figure2](Figure3.png)
*Overview of the project pipeline. (A) Three bilateral acquisitions per muscle, exemplified by Rectus Abdominis, Biceps Brachii, Tibialis Anterior, and Gastrocnemius Medialis. (B) Segmentation: Networks segment and classify muscles, extracting features from the Cross Sectional Area (green) and the deeper muscle region (red) using PyRadiomics. (C) Features are dimensionally reduced by averaging spatially robust metrics. (D) The classification model predicts the probability of the modified Heckmatt class, achieving up to 0.97 AUC for abnormal cases, thereby enhancing diagnostic consistency in neuromuscular disease evaluation, including FSHD.*

The repository is structured as follows:
- **`mmsegmentation/`**: Contains deep learning code for muscle segmentation & classification with K-Net.
- **`feature_extraction/`**: Scripts to extract texture/radiomics features and evaluate segmentation metrics.
- **`prediction_heckmatt/`**: Implements Heckmatt scoring with XGBoost on the extracted radiomics features.


### 1. **`mmsegmentation/`**
Contains the code for muscle segmentation & classification with **K-Net** (based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)):

- **`tools/train.py`**  
  Train your segmentation model (multi-class or muscle-specific).
- **`tools/test.py`**  
  Evaluate a trained model on a test set.
- **`tools/local_inference.py`**  
  Quick local inference for debugging and a more flexible image source than test.py.
- **`utils/compareRevisionResults.py`**  
  Compare segmentation metrics (IoU, precision, recall) for different model versions (multi-label / binary / muscle-specific) with statistical tests.

### 2. **`feature_extraction/`**
Scripts to **extract texture/radiomics features** and evaluate segmentation metrics:

- **`extractNormalizedTextureFeaturesFast.py`**  
  Extracts radiomics features with PyRadiomics, writing to JSON/Excel.  
- **`computeMetricsAndValuesFast.py`**  
  Generates confusion matrices, classification reports, IoU stats, etc.  
- **`readDicom.py`** & **`readSAV.py`**  
  Helpers for reading DICOM or `.sav` data, are not strictly needed if you already have PNG images/tabular data.

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
   - This trains the segmentation network. Edit the config file to train the model in the different modalities (multi-label / binary / muscle-specific).

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
