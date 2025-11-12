function cfg = project_config()
% PROJECT_CONFIG configuration for Urban Infrastructure pipeline
cfg.imgDir = fullfile('dataset','images');
cfg.maskDir = fullfile('dataset','masks');
cfg.outputDir = fullfile('results');
if ~exist(cfg.outputDir,'dir'), mkdir(cfg.outputDir); end
cfg.imgSize = [256 256];
cfg.classes = {'water','road','building','bareland','vegetation'};
% Colors detected from sample masks (RGB)
cfg.colorMap = [
    110 193 228;  % water (light blue)
    132  41 246;  % road (purple)
    155 155 155;  % building (gray)
    226 169  41;  % bareland (orange)
    254 221  58;  % vegetation (yellow)
];
cfg.tolerance = 30; % color matching tolerance
cfg.trainRatio = 0.7;
cfg.valRatio = 0.15;
cfg.testRatio = 0.15;
end
