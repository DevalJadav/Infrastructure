function verify_predictions(images, maskDir, testIdx, knnModel, nbModel, dtModel, config, nShow)
% Displays nShow random test images with true mask and model predictions
if nargin<8, nShow = 6; end
rng(1);
sel = testIdx(randperm(numel(testIdx), min(nShow,numel(testIdx))));
figure('Name','Verification Samples','NumberTitle','off','Units','normalized','Position',[0.1 0.1 0.8 0.6]);
for i=1:numel(sel)
    idx = sel(i);
    I = imread(images{idx}); mask = imread(fullfile(maskDir, sprintf('mask%s.png', regexp(images{idx}, '\d+', 'match'))));
    Iproc = preprocess_image(I, config.imgSize);
    feat = extract_features(Iproc);
    % normalize using training mu/sigma if saved; fallback to raw feature
    % We assume models trained on normalized features; predict using model assuming same scaling done in main
    % Here for simplicity we use knnModel.X features space - works if using same scaling
    pred_knn = predict_knn(knnModel, (feat - mean(knnModel.X,1))./std(knnModel.X,[],1)); %#ok<NASGU>
    pred_nb = predict_naivebayes(nbModel, (feat - mean(knnModel.X,1))./std(knnModel.X,[],1));
    pred_dt = predict_decision_tree(dtModel, (feat - mean(knnModel.X,1))./std(knnModel.X,[],1));
    subplot(2,ceil(numel(sel)/2),i);
    imshow(I); axis off;
    title(sprintf('True: %s\\nKNN:%d NB:%d DT:%d', 'see mask', pred_knn, pred_nb, pred_dt));
end
end
