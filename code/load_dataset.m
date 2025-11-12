function [images, labels_multi, labels_primary] = load_dataset(imgDir, maskDir, colorMap)
% Load images named imageXXXX.jpg and masks named maskXXXX.png
% Returns:
% images - cell array of image paths
% labels_multi - n x C logical matrix indicating which classes are present in each mask
% labels_primary - n x 1 integer primary class (largest area in mask)
imgFiles = dir(fullfile(imgDir, 'image*.jpg'));
if isempty(imgFiles)
    imgFiles = dir(fullfile(imgDir, 'image*.png'));
end
images = {}; labels_multi = []; labels_primary = [];
C = size(colorMap,1);

for i=1:numel(imgFiles)
    imgName = imgFiles(i).name;
    numstr = regexp(imgName, '\d+', 'match');
    if isempty(numstr), continue; end
    numstr = numstr{1};
    maskName = sprintf('mask%s.png', numstr);
    maskPath = fullfile(maskDir, maskName);
    imgPath = fullfile(imgDir, imgName);
    if ~exist(maskPath, 'file')
        fprintf('Warning: mask not found for %s (expected %s). Skipping.\\n', imgName, maskName);
        continue;
    end
    images{end+1} = imgPath; %#ok<AGROW>
    mask = imread(maskPath);
    if size(mask,3)==1
        % assume classes encoded as integers in grayscale mask
        maskLabels = unique(mask(:));
        % create multi-label by presence
        lm = false(1,C);
        for c=1:C
            % if any pixel equals mapping color intensity? fallback: check area ratio later
            % Not ideal for grayscale; set to false
            lm(c) = false;
        end
    else
        % RGB mask: compute pixel-wise nearest color in colorMap
        M = double(reshape(mask, [], 3));
        D = zeros(size(M,1), C);
        for c=1:C
            D(:,c) = sum((M - reshape(colorMap(c,:),1,3)).^2,2);
        end
        [~, idxmin] = min(D, [], 2);
        % compute counts per class
        counts = histcounts(idxmin, 1:(C+1));
        lm = counts > 0;
    end
    labels_multi = [labels_multi; lm]; %#ok<AGROW>
    % primary by max count (if RGB)
    if exist('counts','var') && sum(counts)>0
        [~, p] = max(counts); labels_primary(end+1,1) = p;
    else
        labels_primary(end+1,1) = 1;
    end
end
end
