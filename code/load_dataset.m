function [images, labels] = load_dataset(dataDir, classList, imgSize)
images = {}; labels = [];
for c = 1:numel(classList)
    folder = fullfile(dataDir, classList{c});
    if ~exist(folder,'dir'), continue; end
    files = dir(fullfile(folder, '*.png'));
    if isempty(files), files = dir(fullfile(folder,'*.jpg')); end
    for k=1:numel(files)
        images{end+1} = fullfile(folder, files(k).name);
        labels(end+1,1) = c;
    end
end
perm = randperm(numel(images));
images = images(perm); labels = labels(perm);
end
