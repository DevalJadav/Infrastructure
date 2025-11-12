function [images, labels_multi, labels_primary] = load_dataset(imgDir, maskDir, colorMap, tol)
% LOAD_DATASET Loads image paths and computes multi-label and primary labels
% images: cell array of image full paths
% labels_multi: N x C logical matrix
% labels_primary: N x 1 integer (dominant class)
if nargin<4, tol = 30; end
imgFiles = dir(fullfile(imgDir, 'image*.jpg'));
if isempty(imgFiles), imgFiles = dir(fullfile(imgDir,'*.png')); end
images = {};
labels_multi = [];
labels_primary = [];
C = size(colorMap,1);
for i=1:numel(imgFiles)
    imgName = imgFiles(i).name;
    numstr = regexp(imgName, '\d+', 'match');
    if isempty(numstr), continue; end
    numstr = numstr{1};
    maskName = sprintf('mask%s.png', numstr);
    imgPath = fullfile(imgFiles(i).folder, imgName);
    maskPath = fullfile(maskDir, maskName);
    if ~exist(maskPath,'file')
        fprintf('Warning: mask not found for %s. Skipping.\n', imgName);
        continue;
    end
    images{end+1} = imgPath; %#ok<AGROW>
    mask = imread(maskPath);
    if size(mask,3)==1
        mask = repmat(mask, [1 1 3]);
    end
    M = double(reshape(mask, [], 3));
    D = zeros(size(M,1), C);
    for c=1:C
        D(:,c) = sqrt(sum((M - reshape(colorMap(c,:),1,3)).^2, 2));
    end
    [mind, idxmin] = min(D, [], 2);
    idxmin(mind > tol) = 0; % unclassified pixels
    if all(idxmin==0)
        counts = zeros(1,C);
    else
        counts = histcounts(idxmin(idxmin>0), 1:(C+1));
    end
    lm = counts > 0;
    labels_multi = [labels_multi; lm]; %#ok<AGROW>
    if sum(counts) > 0
        [~, p] = max(counts); labels_primary(end+1,1) = p;
    else
        labels_primary(end+1,1) = 0;
    end
end
end
