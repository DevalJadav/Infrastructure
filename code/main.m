% Main script - Urban Infrastructure Analysis
rng(42); clear; clc;
fprintf('Loading dataset and extracting labels from masks...\n');
cfg = project_config();
[images, labels_multi, labels_primary] = load_dataset(cfg.imgDir, cfg.maskDir, cfg.colorMap, cfg.tolerance);
fprintf('Found %d image-mask pairs\n', numel(images));
verify_dataset_balance(labels_multi, cfg.classes);
counts = sum(labels_multi,1);
for c=1:numel(cfg.classes)
    fprintf('Class %s: %d images (may include multi-labels)\n', cfg.classes{c}, counts(c));
end
if any(counts < 100)
    warning('One or more classes have fewer than 100 images. Consider augmentation or tile sampling.');
end
[trainIdx,valIdx,testIdx] = stratified_split(labels_primary,cfg.trainRatio,cfg.valRatio,cfg.testRatio);
fprintf('Split into %d train, %d val, %d test\n',numel(trainIdx),numel(valIdx),numel(testIdx));
if ~exist('results','dir'), mkdir('results'); end
featFile = fullfile('results','features.mat');
if exist(featFile,'file')
    load(featFile,'X','y_primary','meta','selected','mu','sigma');
    fprintf('Loaded cached features from %s\n', featFile);
else
    fprintf('Extracting features for %d images...\n',numel(images));
    X=[]; meta.filenames=images;
    for i=1:numel(images)
        I=imread(images{i});
        Iproc=preprocess_image(I,cfg.imgSize);
        f=extract_features(Iproc);
        X=[X;f];
        if mod(i,50)==0 || i==numel(images), fprintf('  processed %d/%d\n',i,numel(images)); end
    end
    y_primary=labels_primary;
    selected=1:size(X,2);
    mu=mean(X(trainIdx,:),1); sigma=std(X(trainIdx,:),[],1); sigma(sigma==0)=1;
    save(featFile,'X','y_primary','meta','selected','mu','sigma','-v7.3');
    fprintf('Saved features to %s\n',featFile);
end
fprintf('Performing ANOVA selection...\n');
pValues=anova_feature_selection(X,labels_primary);
selected=find(pValues<0.05);
if isempty(selected)
    warning('No features passed ANOVA threshold; using all features.');
    selected=1:size(X,2);
end
Xsel=X(:,selected);
mu=mean(Xsel(trainIdx,:),1); sigma=std(Xsel(trainIdx,:),[],1); sigma(sigma==0)=1;
Xnorm=(Xsel-mu)./sigma;
Xtrain=Xnorm(trainIdx,:); ytrain=labels_primary(trainIdx);
Xval=Xnorm(valIdx,:); yval=labels_primary(valIdx);
Xtest=Xnorm(testIdx,:); ytest=labels_primary(testIdx);
fprintf('Training KNN (k=5)...\n'); knnModel=train_knn(Xtrain,ytrain,5);
fprintf('Training Naive Bayes...\n'); nbModel=train_naivebayes(Xtrain,ytrain);
fprintf('Training Decision Tree...\n'); dtModel=train_decision_tree(Xtrain,ytrain,20);
fprintf('Predicting on test set...\n');
ypred_knn=predict_knn(knnModel,Xtest);
ypred_nb=predict_naivebayes(nbModel,Xtest);
ypred_dt=predict_decision_tree(dtModel,Xtest);
metrics_knn=evaluate_metrics(ytest,ypred_knn,cfg.classes);
metrics_nb=evaluate_metrics(ytest,ypred_nb,cfg.classes);
metrics_dt=evaluate_metrics(ytest,ypred_dt,cfg.classes);
fprintf('KNN Accuracy: %.2f%%\n',metrics_knn.accuracy*100);
fprintf('NB Accuracy:  %.2f%%\n',metrics_nb.accuracy*100);
fprintf('DT Accuracy:  %.2f%%\n',metrics_dt.accuracy*100);
fprintf('Generating professional visualizations...\n');
visualize_pipeline(images,meta,cfg,Xsel,Xnorm,selected,labels_primary,ypred_knn,ypred_nb,ypred_dt,testIdx);
verify_predictions(images,cfg.maskDir,testIdx,knnModel,nbModel,dtModel,cfg,8);
fprintf('All done. Review figures for professional outputs.\n');
