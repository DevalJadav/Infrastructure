% Main script - Urban Infrastructure Analysis
rng(42); clear; clc;

fprintf('Loading dataset and extracting labels from masks...\n');
cfg = project_config();

% Load images, masks, and labels
[images, masks, labels_multi, labels_primary] = load_dataset( ...
    cfg.imgDir, cfg.maskDir, cfg.colorMap, cfg.tolerance);

fprintf('Found %d image-mask pairs\n', numel(images));

% Check class balance
verify_dataset_balance(labels_multi, cfg.classes);
counts = sum(labels_multi, 1);
for c = 1:numel(cfg.classes)
    fprintf('Class %s: %d images (may include multi-labels)\n', ...
        cfg.classes{c}, counts(c));
end
if any(counts < 100)
    warning('One or more classes have fewer than 100 images. Consider augmentation or tile sampling.');
end

% Stratified split
[trainIdx, valIdx, testIdx] = stratified_split(labels_primary, ...
    cfg.trainRatio, cfg.valRatio, cfg.testRatio);
fprintf('Split into %d train, %d val, %d test\n', ...
    numel(trainIdx), numel(valIdx), numel(testIdx));

% Feature cache handling
if ~exist('results', 'dir')
    mkdir('results');
end
featFile = fullfile('results', 'features.mat');

if exist(featFile, 'file')
    load(featFile, 'X', 'y_primary', 'meta', 'selected', 'mu', 'sigma');
    fprintf('Loaded cached features from %s\n', featFile);
else
    fprintf('Extracting features for %d images...\n', numel(images));
    X = [];
    meta.filenames = images;

    for i = 1:numel(images)
        I = imread(images{i});
        Iproc = preprocess_image(I, cfg.imgSize);
        f = extract_features(Iproc);
        X = [X; f]; %#ok<AGROW>

        if mod(i, 50) == 0 || i == numel(images)
            fprintf('  processed %d/%d\n', i, numel(images));
        end
    end

    y_primary = labels_primary;

    % Initially select all features
    selected = 1:size(X, 2);

    % Compute normalisation stats on training subset (rough initial)
    mu = mean(X(trainIdx, :), 1);
    sigma = std(X(trainIdx, :), 0, 1);
    sigma(sigma == 0) = 1;

    save(featFile, 'X', 'y_primary', 'meta', 'selected', 'mu', 'sigma', '-v7.3');
    fprintf('Saved features to %s\n', featFile);
end

% ANOVA feature selection
fprintf('Performing ANOVA selection...\n');
pValues = anova_feature_selection(X, labels_primary);
selected = find(pValues < 0.05);
if isempty(selected)
    warning('No features passed ANOVA threshold; using all features.');
    selected = 1:size(X, 2);
end

Xsel = X(:, selected);

% Recompute normalisation on selected features
mu = mean(Xsel(trainIdx, :), 1);
sigma = std(Xsel(trainIdx, :), 0, 1);
sigma(sigma == 0) = 1;

% Manually expand mu and sigma to match Xsel (avoid broadcasting issues)
muMat    = repmat(mu,    size(Xsel, 1), 1);
sigmaMat = repmat(sigma, size(Xsel, 1), 1);

Xnorm = (Xsel - muMat) ./ sigmaMat;

Xtrain = Xnorm(trainIdx, :);  ytrain = labels_primary(trainIdx);
Xval   = Xnorm(valIdx, :);    yval   = labels_primary(valIdx);
Xtest  = Xnorm(testIdx, :);   ytest  = labels_primary(testIdx);

% Train models
fprintf('Training KNN (k=5)...\n');
knnModel = train_knn(Xtrain, ytrain, 5);

fprintf('Training Naive Bayes...\n');
nbModel = train_naivebayes(Xtrain, ytrain);

fprintf('Training Decision Tree...\n');
dtModel = train_decision_tree(Xtrain, ytrain, 20);

% Predict on test set
fprintf('Predicting on test set...\n');
ypred_knn = predict_knn(knnModel, Xtest);
ypred_nb  = predict_naivebayes(nbModel, Xtest);
ypred_dt  = predict_decision_tree(dtModel, Xtest);

% Evaluate
metrics_knn = evaluate_metrics(ytest, ypred_knn, cfg.classes);
metrics_nb  = evaluate_metrics(ytest, ypred_nb,  cfg.classes);
metrics_dt  = evaluate_metrics(ytest, ypred_dt,  cfg.classes);

fprintf('KNN Accuracy: %.2f%%\n', metrics_knn.accuracy * 100);
fprintf('NB Accuracy:  %.2f%%\n', metrics_nb.accuracy  * 100);
fprintf('DT Accuracy:  %.2f%%\n', metrics_dt.accuracy  * 100);

% Confusion matrix of best model (Decision Tree here)
K = numel(cfg.classes);

% Remove invalid labels (0) from evaluation for confusion matrix
valid = (ytest >= 1) & (ytest <= K);
ytrue_valid = ytest(valid);
ypred_valid = ypred_dt(valid);

Cbest = confusionmat(ytrue_valid, ypred_valid, 'Order', 1:K);

% Professional visualisations (saved as PNGs)
fprintf('Generating professional visualizations...\n');
visualize_pipeline(images,masks, labels_primary, Xnorm, cfg.classes,Cbest);                   % confusion matrix of best model

% Optional: verification visualisation of predictions vs masks
verify_predictions(images, cfg.maskDir, testIdx, knnModel, nbModel, dtModel, cfg, 8);

fprintf('All done. Review figures in results/figures for professional outputs.\n');
