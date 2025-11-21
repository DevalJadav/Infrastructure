function [images, masks, labels_multi, labels_primary] = load_dataset(imgDir, maskDir, colorMap, tol)
% LOAD_DATASET Loads image paths, mask paths and computes multi-label and primary labels
%
% Outputs:
%   images        : 1xN cell array of image full paths
%   masks         : 1xN cell array of mask full paths
%   labels_multi  : N x C logical matrix (multi-label flags per class)
%   labels_primary: N x 1 integer (dominant class index, 0 = none)
%
% Inputs:
%   imgDir   : directory containing image files
%   maskDir  : directory containing mask files
%   colorMap : C x 3 matrix of RGB class colours
%   tol      : distance tolerance for class assignment (default = 30)

    if nargin < 4
        tol = 30;
    end

    imgFiles = dir(fullfile(imgDir, 'image*.jpg'));
    if isempty(imgFiles)
        imgFiles = dir(fullfile(imgDir, '*.png'));
    end

    images         = {};
    masks          = {};
    labels_multi   = [];
    labels_primary = [];
    C = size(colorMap, 1);  % number of classes

    for i = 1:numel(imgFiles)
        imgName = imgFiles(i).name;

        % Extract numeric ID from image file name (e.g. image123.jpg -> "123")
        numstr = regexp(imgName, '\d+', 'match');
        if isempty(numstr)
            continue;
        end
        numstr = numstr{1};

        % Corresponding mask name: maskXXX.png
        maskName = sprintf('mask%s.png', numstr);

        imgPath  = fullfile(imgFiles(i).folder, imgName);
        maskPath = fullfile(maskDir, maskName);

        if ~exist(maskPath, 'file')
            fprintf('Warning: mask not found for %s. Skipping.\n', imgName);
            continue;
        end

        % Store paths
        images{end+1} = imgPath;   %#ok<AGROW>
        masks{end+1}  = maskPath;  %#ok<AGROW>

        % Read mask to compute labels
        mask = imread(maskPath);
        if size(mask, 3) == 1
            mask = repmat(mask, [1 1 3]);
        end

        % Flatten mask to N x 3 for distance computation
        M = double(reshape(mask, [], 3));  % each row = [R G B]

        % Compute distance from each pixel to each class color
        D = zeros(size(M, 1), C);
        for c = 1:C
            D(:, c) = sqrt(sum((M - reshape(colorMap(c, :), 1, 3)).^2, 2));
        end

        [mind, idxmin] = min(D, [], 2);

        % Assign only if within tolerance
        idxmin(mind > tol) = 0;  % 0 = unclassified pixel

        if all(idxmin == 0)
            counts = zeros(1, C);
        else
            counts = histcounts(idxmin(idxmin > 0), 1:(C+1));
        end

        lm = counts > 0;              % multi-label flags
        labels_multi = [labels_multi; lm]; %#ok<AGROW>

        if sum(counts) > 0
            [~, p] = max(counts);
            labels_primary(end+1, 1) = p; %#ok<AGROW>
        else
            labels_primary(end+1, 1) = 0; %#ok<AGROW>
        end
    end
end
