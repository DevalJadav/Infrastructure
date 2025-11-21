function visualize_pipeline(images, masks, labels_primary, Xnorm, classNames, Cbest)
% VISUALIZE_PIPELINE
%   Creates all figures needed for the report/presentation:
%   1) Sample image + mask (side by side)
%   2) Boxplot of one feature by class
%   3) 2D scatter of two features
%   4) Confusion matrix of best model
%
% Inputs:
%   images        : 1xN cell array of image file paths
%   masks         : 1xN cell array of mask file paths
%   labels_primary: Nx1 numeric labels (0..K), 0 = unclassified
%   Xnorm         : NxD normalized feature matrix
%   classNames    : 1xK cell array of class names
%   Cbest         : KxK confusion matrix for best model

    % ---------- OUTPUT DIRECTORY (ABSOLUTE PATH) ----------
    % Save figures relative to this file location: ../results/figures
    thisFileDir = fileparts(mfilename('fullpath'));
    outDir = fullfile(thisFileDir, '..', 'results', 'figures');
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    fprintf('Visuals will be saved to: %s\n', outDir);

    % ============ 1. SAMPLE IMAGE + MASK ============
    sampleIdx = min(10, numel(images));  % safe index
    img  = imread(images{sampleIdx});
    mask = imread(masks{sampleIdx});
    f2 = visualize_sample_image_and_mask(img, mask);
    save_fig(f2, outDir, 'sample_image_and_mask');

    % ============ 2. FEATURE BOXPLOT ============
    featureIndex = 1;                 % adjust to real feature (e.g. Mean Green)
    featureName  = 'Feature 1';       % rename in report if needed
    f3 = visualize_feature_boxplot(Xnorm, labels_primary, classNames, featureIndex, featureName);
    save_fig(f3, outDir, 'feature_boxplot');

    % ============ 3. 2D FEATURE SCATTER ============
    if size(Xnorm, 2) >= 2
        idx1  = 1;
        idx2  = 2;
        name1 = 'Feature 1';
        name2 = 'Feature 2';
        f4 = visualize_feature_scatter_2d(Xnorm, labels_primary, classNames, idx1, idx2, name1, name2);
        save_fig(f4, outDir, 'feature_scatter_2d');
    else
        fprintf('Skipping 2D scatter: not enough feature dimensions.\n');
    end

    % ============ 4. CONFUSION MATRIX ============
    f5 = visualize_confusion_matrix(Cbest, classNames, 'Best Model Confusion Matrix');
    save_fig(f5, outDir, 'confusion_matrix_best_model');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = visualize_sample_image_and_mask(img, mask)
    % White background per figure
    f = figure('Name','Sample Image and Mask','NumberTitle','off', ...
               'Color','w');
    subplot(1,2,1);
    imshow(img, 'Border','tight');
    title('Original Satellite Image');

    subplot(1,2,2);
    imshow(mask, 'Border','tight');
    title('Segmentation Mask');
end

function f = visualize_feature_boxplot(features, labelsNumeric, classNames, idx, featureName)
% VISUALIZE_FEATURE_BOXPLOT
%   Draws a boxplot of one feature grouped by class.
%   Ignores samples with label 0 (unclassified).
%   Uses default group labels and sets x-tick labels manually.

    K = numel(classNames);

    % Keep only valid labels 1..K
    valid = (labelsNumeric >= 1) & (labelsNumeric <= K);

    fVals = features(valid, idx);
    gVals = labelsNumeric(valid);

    fVals = fVals(:);
    gVals = gVals(:);

    f = figure('Name',['Feature Boxplot: ' featureName], ...
               'NumberTitle','off', 'Color','w');

    boxplot(fVals, gVals);
    grid on;
    xlabel('Class');
    ylabel(featureName);
    title(['Distribution of ', featureName, ' across classes']);

    ax = gca;
    ax.XTick = 1:K;
    ax.XTickLabel = classNames;
end

function f = visualize_feature_scatter_2d(features, labelsNumeric, classNames, idx1, idx2, name1, name2)
    f = figure('Name','2D Feature Scatter','NumberTitle','off', ...
               'Color','w');
    hold on;
    K = numel(classNames);

    for k = 1:K
        mask = (labelsNumeric == k);
        scatter(features(mask, idx1), features(mask, idx2), 30, 'filled', ...
            'DisplayName', classNames{k});
    end

    xlabel(name1);
    ylabel(name2);
    title('2D Feature Scatter Plot');
    legend('Location','bestoutside');
    grid on;
    hold off;
end

function f = visualize_confusion_matrix(C, classNames, titleStr)
    f = figure('Name','Confusion Matrix','NumberTitle','off', ...
               'Color','w');
    imagesc(C);
    colormap('turbo');   % higher contrast on white
    colorbar;

    axis equal tight;
    xticks(1:numel(classNames));
    yticks(1:numel(classNames));
    xticklabels(classNames);
    yticklabels(classNames);
    xlabel('Predicted');
    ylabel('True');
    title(titleStr);

    for i = 1:size(C,1)
        for j = 1:size(C,2)
            text(j, i, num2str(C(i,j)), ...
                'HorizontalAlignment','center', ...
                'Color','w', 'FontWeight','bold');
        end
    end
end

function save_fig(figHandle, outDir, filename)
% SAVE_FIG Save figure as high-quality PNG (300 DPI) with clean background
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    filepath = fullfile(outDir, [filename '.png']);
    fprintf('Saving figure to: %s\n', filepath);

    % exportgraphics gives smoother, better-looking images than print
    try
        exportgraphics(figHandle, filepath, ...
            'BackgroundColor','white', ...
            'Resolution',300);
    catch
        % Fallback for older MATLAB versions
        set(figHandle, 'Color','white');
        set(figHandle, 'InvertHardcopy','off');
        print(figHandle, filepath, '-dpng', '-r300');
    end
end
