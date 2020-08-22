root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

bboxes_root = fullfile(DATA_ROOT, bboxes_imgset);
mixed = load(fullfile(root, 'mixed/mixed_lite.mat'));

class_indices = get_class_indices(ROOT, imgset);

for cl = 1:numel(classes)-1
  clname = classes{cl};
  fprintf('Processing for class %s\n', clname);
  imdb.proposals = mixed.proposals(class_indices{cl});
  imdb.images_size = mixed.images_size(class_indices{cl});
  imdb.root_ids = mixed.root_ids(class_indices{cl});
  if isfield(mixed, 'root_feat_types')
    imdb.root_feat_types = mixed.root_feat_types(class_indices{cl});
  end
  if isfield(mixed, 'root_feat_types_code')
    imdb.root_feat_types_code = mixed.root_feat_types_code(class_indices{cl});
  end

  imdb.bboxes = getfield(load(fullfile(bboxes_root, clname, [clname, '_lite.mat'])), 'bboxes');
  
  mkdir(fullfile(root, clname));
  save_path = fullfile(root, clname, [clname, '_lite.mat']);
  savefile(save_path, imdb); 
end
fprintf('\n');