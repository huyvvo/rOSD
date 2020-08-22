% save 'bboxes' and 'proposals' as separate imdbs to speed up
% loading when images are not required. Also, an additional field
% 'image_size' will also be added.


root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

fprintf('Create lite imdb for class ')
for cl = 1:numel(classes)
  clname = classes{cl};
  fprintf('%s ', clname);
  imdb_path = fullfile(root, clname, [clname, '.mat']);
  save_path = fullfile(root, clname, [clname, '_lite.mat']);
  imdb = load(imdb_path);
  n = size(imdb.images, 1);
  images_size = cellfun(@(x) [size(x,1), size(x,2)], imdb.images, 'UniformOutput', false);
  imdb = rmfield(imdb, 'images');
  imdb.images_size = images_size;
  save(save_path, '-struct', 'imdb');
end
fprintf('\n');