% Main script - Professional Final Pipeline
% Produces enhanced visualizations, dataset checks, and verification outputs
rng(42); clear; clc;

% === Configuration ===
config.imgDir = 'dataset/images';    % images like image0001.jpg
config.maskDir = 'dataset/masks';    % masks like mask0001.png
config.imgSize = [256,256];
config.classes = {'roads','buildings','vegetation','water','bareland'};
% colorMap rows correspond to classes above - adjust to your mask colors
config.colorMap = [
    128 64 128;   % roads
    70 70 70;     % buildings
    107 142 35;   % vegetation
    0 0 255;      % water
    150 150 150;  % bareland
    ];

config.trainRatio = 0.70;
config.valRatio = 0.15;
config.testRatio = 0.15;

% === Load dataset (multi-label aware) ===
fprintf('Loading dataset and extracting labels from masks...\n');
[images, labels_multi, labels_primary] = load_dataset(config.imgDir, config.maskDir, config.colorMap);
n = numel(images);
fprintf('Found %d image-mask pairs\\n', n);

% === Verify dataset balance ===
verify_dataset_balance(labels_multi, config.classes);

% Check that each class has at least 100 images (as required)
counts = sum(labels_multi,1);
for c=1:numel(config.classes)
    fprintf('Class %s: %d images (may include multi-labels)\n', config.classes{c}, counts(c));
end
if any(counts < 100)
    warning('One or more classes have fewer than 100 images. Consider augmentation or tile sampling.');
end

% === Split (stratified by primary label) ===
[trainIdx, valIdx, testIdx] = stratified_split(labels_primary, config.trainRatio, config.valRatio, config.testRatio);
fprintf('Split into %d train, %d val, %d test\\n', numel(trainIdx), numel(valIdx), numel(testIdx));

% === Feature extraction & cache ===
if ~exist('results','dir'), mkdir('results'); end
featFile = fullfile('results','features.mat');
if exist(featFile,'file')
    load(featFile,'X','y_primary','meta','selected','mu','sigma');
    fprintf('Loaded cached features (%s)\\n', featFile);
else
    fprintf('Extracting features for %d images...\\n', n);
    X = []; meta.filenames = images;
    for i=1:n
        I = imread(images{i});
        Iproc = preprocess_image(I, config.imgSize);
        f = extract_features(Iproc);
        X = [X; f]; %#ok<AGROW>
        if mod(i,50)==0 || i==n, fprintf('  processed %d/%d\\n', i, n); end
    end
    y_primary = labels_primary;
    selected = 1:size(X,2);
    mu = mean(X(trainIdx,:),1); sigma = std(X(trainIdx,:),[],1); sigma(sigma==0)=1;
    save(featFile,'X','y_primary','meta','selected','mu','sigma','-v7.3');
    fprintf('Saved features to %s\\n', featFile);
end

% === Feature selection via ANOVA ===
fprintf('Performing ANOVA selection...\\n');
pValues = anova_feature_selection(X,y_primary);
selected = find(pValues < 0.05);
if isempty(selected)
    warning('No features passed ANOVA threshold; using all features.');
    selected = 1:size(X,2);
end
Xsel = X(:,selected);

% === Standardize ===
mu = mean(Xsel(trainIdx,:),1); sigma = std(Xsel(trainIdx,:),[],1); sigma(sigma==0)=1;
Xnorm = (Xsel - mu) ./ sigma;

Xtrain = Xnorm(trainIdx,:); ytrain = y_primary(trainIdx);
Xval = Xnorm(valIdx,:); yval = y_primary(valIdx);
Xtest = Xnorm(testIdx,:); ytest = y_primary(testIdx);

% === Train models ===
fprintf('Training KNN (k=5)...\\n');
knnModel = train_knn(Xtrain,ytrain,5);

fprintf('Training Naive Bayes...\\n');
nbModel = train_naivebayes(Xtrain,ytrain);

fprintf('Training Decision Tree (fitctree)...\\n');
dtModel = train_decision_tree(Xtrain,ytrain,20);

% === Predictions ===
fprintf('Predicting on test set...\\n');
ypred_knn = predict_knn(knnModel, Xtest);
ypred_nb = predict_naivebayes(nbModel, Xtest);
ypred_dt = predict_decision_tree(dtModel, Xtest);

% === Evaluate ===
metrics_knn = evaluate_metrics(ytest, ypred_knn, config.classes);
metrics_nb = evaluate_metrics(ytest, ypred_nb, config.classes);
metrics_dt = evaluate_metrics(ytest, ypred_dt, config.classes);

fprintf('KNN Accuracy: %.2f%%\\n', metrics_knn.accuracy*100);
fprintf('NB Accuracy:  %.2f%%\\n', metrics_nb.accuracy*100);
fprintf('DT Accuracy:  %.2f%%\\n', metrics_dt.accuracy*100);

% === Professional visualization of pipeline and results ===
visualize_pipeline(images, meta, config, Xsel, Xnorm, selected, y_primary, ypred_knn, ypred_nb, ypred_dt, testIdx);

% === Verification: show random sample predictions with masks ===
verify_predictions(images, config.maskDir, testIdx, knnModel, nbModel, dtModel, config, 8);

fprintf('All done. Review figures for professional outputs.\\n');
