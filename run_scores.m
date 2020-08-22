% This script computes confidence score on a single CPU.
% One can modify it by changing 'class_indices' and 'row_indices'
% to divide the computation into multiple jobs and run on multiple
% CPUs.

cd(fullfile(ROOT, 'commons/scores'));
addpath(genpath(fullfile(ROOT, 'commons')));
imgset = 'vocx_cnn';
classes = get_classes(imgset);

neighbor_imgset = 'vocx';
neighbor_type = 'neighbor_cos_vgg19_fc6_matconvnet_resize';
num_neighbors = 50;

num_images = [];
for cl = 1:numel(classes)
  imdb = load(fullfile(DATA_ROOT, imgset, classes{cl}, [classes{cl}, '_lite.mat']), 'bboxes');
  num_images(end+1) = numel(imdb.bboxes);
end

all_classes = 1:numel(classes);
big_classes = unique([find(num_images > 500), numel(classes)]);

for cl = 1:numel(classes)
  class_indices = cl;
  if ~ismember(cl, big_classes)
    row_indices = 1:num_images(cl)*(num_images(cl)-1)/2;
  else
    e = getfield(load(fullfile(DATA_ROOT, neighbor_imgset, neighbor_type, classes{cl}, ...
                               sprintf('%d.mat', num_neighbors))), 'e');
    actual_num_nb = cellfun(@numel, e);
    indices = [repelem([1:num_images(cl)]', actual_num_nb) reshape(cell2mat(e'), [], 1)];
    indices = [min(indices'); max(indices')]';
    indices = unique(indices, 'rows');
    row_indices = 1:size(indices, 1);
  end
  script;
end

cd(fullfile(ROOT, 'commons/scores/gather_score'));
row_indices = 1:numel(classes);
gather_score;