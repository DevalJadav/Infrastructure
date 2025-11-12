MATLAB Urban Infrastructure - Professional Final Code
=====================================================

Contents:
- main.m : orchestrates the full pipeline
- load_dataset.m : loads image/mask pairs, computes multi-label and primary label
- preprocess_image.m : resizing, denoising, normalization
- extract_features.m : computes GLCM-based and other features
- compute_glcm_stats.m : helper for GLCM stats
- edge_detect.m : Sobel-like edge detection
- anova_feature_selection.m : feature selection via ANOVA
- train_knn.m / predict_knn.m : KNN model functions
- train_naivebayes.m / predict_naivebayes.m : Naive Bayes functions
- train_decision_tree.m / predict_decision_tree.m : Decision Tree (uses fitctree)
- evaluate_metrics.m : compute confusion + metrics
- visualize_pipeline.m : professional visualizations for the report
- verify_predictions.m : shows model predictions on random test images
- verify_dataset_balance.m : shows class counts bar chart

Setup:
1. Place your images in Dataset/images named like image0001.jpg
2. Place corresponding masks in Dataset/masks named like mask0001.png
3. Adjust color map in main.m (config.colorMap) to match mask colors if necessary.

Run:
- Start MATLAB, set current folder to project root, and run:
    >> main
- Figures will be created showing preprocessing, features, PCA, confusion matrices, and verification samples.

Notes:
- The pipeline uses a primary label (largest mask area) for training classical single-label classifiers.
- Multi-label information is computed and used for dataset checks; if you need full multi-label classification, modify trainers accordingly.
- The code attempts to avoid heavy toolbox dependencies; however, fitctree is used for decision tree (Statistics and Machine Learning Toolbox). If unavailable, replace with the included custom tree implementation (not provided here).

Contact: Deval - modify as needed for your environment.
