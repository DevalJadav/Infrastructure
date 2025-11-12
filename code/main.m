% ===============================================================
% Main script for Urban Infrastructure Extraction from Satellite Images
% ===============================================================
rng(42); clear; clc;

config.dataDir = 'data/processed';
config.imgSize = [256,256];
config.classes = {'roads','buildings','vegetation','water','bareland'};
config.trainRatio = 0.7;
config.valRatio = 0.15;
config.testRatio = 0.15;

fprintf('Loading dataset...\n');
[images, labels] = load_dataset(config.dataDir, config.classes, config.imgSize);

[trainIdx, valIdx, testIdx] = stratified_split(labels, config.trainRatio, config.valRatio, config.testRatio);

fprintf('Extracting features...\n');
featuresFile = fullfile('results','features.mat');
if exist(featuresFile,'file')
    load(featuresFile,'X','y','meta');
else
    n = numel(images);
    sampleFeat = extract_features(imread(images{1}));
    featLen = numel(sampleFeat);
    X = zeros(n, featLen); y = labels(:); meta = struct('filenames',[]);
    for i=1:n
        img = imread(images{i});
        img = preprocess_image(img, config.imgSize);
        X(i,:) = extract_features(img);
        meta.filenames{i} = images{i};
        if mod(i,50)==0, fprintf('Processed %d/%d images\n', i, n); end
    end
    save(featuresFile,'X','y','meta','-v7.3');
end

fprintf('Feature selection using ANOVA...\n');
pValues = anova_feature_selection(X,y);
selected = find(pValues < 0.05);
if isempty(selected), selected = 1:size(X,2); end
Xsel = X(:,selected);

mu = mean(Xsel(trainIdx,:),1);
sigma = std(Xsel(trainIdx,:),[],1); sigma(sigma==0)=1;
Xnorm = (Xsel - mu) ./ sigma;

Xtrain = Xnorm(trainIdx,:); ytrain = y(trainIdx);
Xval = Xnorm(valIdx,:); yval = y(valIdx);
Xtest = Xnorm(testIdx,:); ytest = y(testIdx);

fprintf('Training models...\n');
knnModel = train_knn(Xtrain,ytrain,5);
nbModel = train_naivebayes(Xtrain,ytrain);
dtModel = train_decision_tree(Xtrain,ytrain,10);

fprintf('Evaluating models...\n');
ypred_knn = predict_knn(knnModel,Xtest);
ypred_nb = predict_naivebayes(nbModel,Xtest);
ypred_dt = predict_decision_tree(dtModel,Xtest);

metrics_knn = evaluate_metrics(ytest, ypred_knn, config.classes);
metrics_nb = evaluate_metrics(ytest, ypred_nb, config.classes);
metrics_dt = evaluate_metrics(ytest, ypred_dt, config.classes);

fprintf('KNN Accuracy: %.2f%%\n', metrics_knn.accuracy*100);
fprintf('Naive Bayes Accuracy: %.2f%%\n', metrics_nb.accuracy*100);
fprintf('Decision Tree Accuracy: %.2f%%\n', metrics_dt.accuracy*100);

visualize_results(Xtest, ytest, ypred_knn, config.classes);
fprintf('Pipeline finished.\n');
