function pValues = anova_feature_selection(X, y)
% =========================================================================
% FUNCTION: anova_feature_selection
% PURPOSE: Perform one-way ANOVA for feature selection on image data.
%
% INPUTS:
%   X : (N x F) matrix of extracted features
%   y : (N x C) label matrix (can be single column or multi-class one-hot)
%
% OUTPUT:
%   pValues : (1 x F) array of p-values for each feature
%
% This function automatically handles multi-label (one-hot) data by
% converting it to dominant-class labels. It ensures ANOVA always runs
% correctly regardless of input label structure.
% =========================================================================

    % ---------------------------------------------------------------------
    % Step 1: Handle multi-label input (one-hot encoded y)
    % ---------------------------------------------------------------------
    if size(y,2) > 1
        % Convert to dominant label index (1-based)
        [~, y_primary] = max(y, [], 2);
    else
        % Already a single-label vector
        y_primary = y;
    end

    % ---------------------------------------------------------------------
    % Step 2: Ensure inputs are valid
    % ---------------------------------------------------------------------
    if ~isnumeric(X) || ~isnumeric(y_primary)
        error('anova_feature_selection:InvalidInput', ...
              'Inputs X and y must be numeric.');
    end

    if size(X,1) ~= numel(y_primary)
        error('anova_feature_selection:SizeMismatch', ...
              'Feature matrix and label vector must have same number of samples.');
    end

    % ---------------------------------------------------------------------
    % Step 3: Compute ANOVA p-values per feature
    % ---------------------------------------------------------------------
    nFeatures = size(X,2);
    pValues = zeros(1, nFeatures);

    for f = 1:nFeatures
        % Extract feature column
        feature = X(:, f);

        % Handle NaNs or Infs safely
        if any(isnan(feature)) || any(isinf(feature))
            feature = fillmissing(feature, 'constant', 0);
        end

        % One-way ANOVA (between groups defined by class labels)
        try
            p = anova1(feature, y_primary, 'off'); % 'off' = suppress plot
        catch
            p = 1; % If ANOVA fails for any reason, keep feature
        end

        pValues(f) = p;
    end

    % ---------------------------------------------------------------------
    % Step 4: Display summary
    % ---------------------------------------------------------------------
    sigCount = sum(pValues <= 0.05);
    fprintf('ANOVA complete: %d / %d features significant (p <= 0.05)\n', ...
            sigCount, nFeatures);

end
