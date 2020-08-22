root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);
class_indices = get_class_indices(ROOT, imgset);

featpath_mixed = fullfile(root, 'mixed', feat_type);

fprintf('Copying features %s\n', feat_type);
for cl = 1:numel(classes)-1
  clname = classes{cl};
  fprintf('Copying for class %s: ', clname);
  featpath_class = fullfile(root, clname, feat_type);
  mkdir(featpath_class)
  n = numel(class_indices{cl});
  for i = 1:n 
    if mod(i, 100) == 1
      fprintf('%d ', i);
    end
    system(sprintf('cp %s %s', fullfile(featpath_mixed, sprintf('%d.mat', class_indices{cl}(i))), ...
                               fullfile(featpath_class, sprintf('%d.mat', i))));
  end
  fprintf('\n');
end