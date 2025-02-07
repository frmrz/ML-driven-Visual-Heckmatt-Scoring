
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
@article{marzola2024machine,
  title={Machine Learning-driven Heckmatt Grading on an FSHD cohort:
         A Novel Pathway for Musculoskeletal Ultrasound Analysis},
  author={Marzola, Francesco and ...},
  journal={Mendeley Data / Preprint / ...},
  year={2024}
}
