Machine learning-driven Heckmatt grading in facioscapulohumeral muscular dystrophy: A novel pathway for musculoskeletal ultrasound analysis

This repository accompanies the paper:

    Machine Learning-driven Heckmatt Grading on an FSHD cohort: A Novel Pathway for Musculoskeletal Ultrasound Analysis
    Francesco Marzola et al., 2024
    Link to Data & Code DOI

It contains the code needed to segment skeletal muscles in ultrasound images using a deep learning approach (K-Net), extract radiomics features from those segments, and perform automated Heckmatt scoring with XGBoost. The final classification step is explained with SHAP for feature-level interpretability.

Below you will find instructions and an overview of each directory so you can reproduce or adapt the pipeline and validate our results.
Repository Structure

.
├── feature_extraction
│   ├── computeMetricsAndValuesFast.py
│   ├── extractNormalizedTextureFeaturesFast.py
│   ├── readDicom.py
│   └── readSAV.py
├── mmsegmentation
│   ├── mmseg
│   │   ├── __init__.py
│   │   └── version.py
│   ├── tools
│   │   ├── local_inference.py
│   │   ├── test.py
│   │   └── train.py
│   ├── utils
│   │   └── compareRevisionResults.py
│   └── README.md
└── prediction_heckmatt
    ├── PredictHeckmattScore_XG_plus_shap_onlyCSA.py
    ├── PredictHeckmattScore_XG_plus_shap_onlyNOT.py
    └── PredictHeckmattScore_XG_plus_shap.py

1. mmsegmentation/

Contains the deep learning code for muscle segmentation & classification using K-Net (a method built upon the MMSegmentation framework). Below there are the codes that have been added or modified with respect to the original repository.

    tools/train.py – script to train your segmentation model (multi-class or muscle-specific K-Net).
    tools/test.py – script to test the trained model on a held-out dataset, producing predictions and evaluation metrics.
    tools/local_inference.py – utility for quick local inference on single images, optional for debugging or demonstration.
    utils/compareRevisionResults.py – compares different trained models or segmentation variants. It runs statistical tests on metrics (IoU, precision, recall) and can produce boxplots/confusion matrices.

2. feature_extraction/

Scripts to extract texture/radiomics features from ultrasound images, evaluate segmentation metrics, or handle DICOM data.

    extractNormalizedTextureFeaturesFast.py – the primary script for computing radiomics features with PyRadiomics. It reads segmented images (either from ground truth or from your model), extracts 1st–2nd–higher-order features (GLCM, GLRLM, etc.), and saves them in a JSON/Excel summary.
    computeMetricsAndValuesFast.py – generates confusion matrices, classification reports, IoU stats for your segmentation results. Also organizes data in Excel for analysis.
    readDicom.py / readSAV.py – examples for reading medical images (DICOM) or stats (SAV) but are not strictly needed for the pipeline if you already have PNG images and tabular data.

3. prediction_heckmatt/

Implements Heckmatt scoring with XGBoost on the extracted radiomics features. Includes scripts to isolate the CSA features, the deeper region, or combine both.

    PredictHeckmattScore_XG_plus_shap.py – trains a 3-class model (Normal/Uncertain/Abnormal) using both CSA and deeper region features; includes SHAP interpretability.
    PredictHeckmattScore_XG_plus_shap_onlyCSA.py – ablation focusing only on the CSA region.
    PredictHeckmattScore_XG_plus_shap_onlyNOT.py – ablation focusing only on the deeper region.
    All scripts output confusion matrices, ROC curves (OvR & OvO), classification reports, correlation with manual z-scores, plus SHAP beeswarm/decision plots.

How to Reproduce the Results

    Install Dependencies & Environment
        Python ≥ 3.8 recommended.
        Required packages: [PyTorch], [MMCV], [MMEngine], [MMSegmentation], [PyRadiomics], [XGBoost], [SHAP], [seaborn/pandas], etc.
        The mmsegmentation folder is a local copy or partial copy of MMSegmentation for convenience. Check your versions of mmcv and mmengine to match those specified in mmseg/__init__.py.

    Prepare the Ultrasound Dataset
        The data used in the paper is shared at Mendeley: Link to Data & Code DOI.
        Split images into train/test folds, or replicate the 5-fold/10-fold cross-validation as in the paper.
        Label your images with muscle IDs if you aim for multi-class K-Net training.

    Run Segmentation Training
        Navigate to mmsegmentation/tools/.
        Modify or create a config for K-Net (or use existing “knet_swin_mod” style configs).
        Example usage:

    python train.py /path/to/your_config.py \
      --work-dir /path/to/save/checkpoints

    This trains the multi-muscle model. The fine-tuning for each muscle is similar, but you specify a muscle-specific config.

Evaluate & Generate Segmentation Maps

    Use test.py to run inference on a held-out fold or test set:

    python test.py /path/to/your_config.py /path/to/checkpoint.pth

Extract Radiomics Features

    In feature_extraction/, edit extractNormalizedTextureFeaturesFast.py to point to your predicted segmentation masks and raw ultrasound images.
    Run:

    python extractNormalizedTextureFeaturesFast.py

    The script outputs a summary Excel/JSON with first-order and second-order features.
    Check the config inside the script to ensure you’re extracting the same features and saving to the same paths as in the paper.

Heckmatt Classification

    Go to prediction_heckmatt/ and pick your desired script:
        PredictHeckmattScore_XG_plus_shap.py to replicate the combined CSA+deeper approach.
        PredictHeckmattScore_XG_plus_shap_onlyCSA.py or _onlyNOT.py for ablation.
    Make sure the script can read your features from the prior step, and that you have columns for “manual_h_score” in your data.
    It will run a cross-validation loop (10-fold) with XGBoost, and produce confusion matrices, ROC curves, and SHAP plots:

        python PredictHeckmattScore_XG_plus_shap.py

        Check outputs in the results directory for correlation with z-scores, classification metrics, etc.

With these steps, you can fully recreate the pipeline and match the results from the paper.
Key Results & Correspondence

    Segmentation
        IoU per muscle is in “Table 2” of the paper. You can replicate that with compareRevisionResults.py or by storing results in CSV/Excel from computeMetricsAndValuesFast.py.
    Classification
        The multi-class muscle classification accuracy also logs from computeMetricsAndValuesFast.py.
    Heckmatt
        Confusion matrix & ROC for (Normal, Uncertain, Abnormal): see scripts in prediction_heckmatt/.
        Ablation: _onlyCSA.py or _onlyNOT.py replicate Table 3.
        SHAP beeswarm/decision plots replicate Fig. 6.

Citation & License

If you use this pipeline, please cite our paper:

@article{marzola2024machine,
  title={Machine Learning-driven Heckmatt Grading on an FSHD cohort:
         A Novel Pathway for Musculoskeletal Ultrasound Analysis},
  author={Marzola, Francesco and ...},
  journal={Mendeley Data / Preprint / ...},
  year={2024}
}

This repository is released under the Apache 2.0 license (see LICENSE in mmsegmentation/), in line with the MMSegmentation license.
Questions or Issues

    For segmentation code or environment specifics, see the original MMSegmentation docs.
    For feature extraction quirks, see PyRadiomics documentation.
    If you find a bug or have a feature request, please open an issue or submit a PR.
