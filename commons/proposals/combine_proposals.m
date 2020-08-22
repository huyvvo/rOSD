root = fullfile(DATA_ROOT, imgset);
verbose = true;

for cl = numel(classes)
  clname = classes{cl};
  save_path = fullfile(root, clname, [clname, '_lite.mat']);
  imdb_paths = {fullfile(root, clname, [clname, '_vgg19_relu44_matconvnet_lite.mat']); ...
                fullfile(root, clname, [clname, '_vgg19_relu54_matconvnet_lite.mat']); ...
                };

  imdbs = {};
  for imdb_idx = 1:numel(imdb_paths)
    imdbs{imdb_idx} = load(imdb_paths{imdb_idx});
  end
  assert(all(cellfun(@(x) isequal(imdbs{1}.bboxes, x.bboxes), imdbs)));
  n = size(imdbs{1}.bboxes, 1);

  fields = fieldnames(imdbs{1});
  imdb = struct;
  for fidx = 1:numel(fields)
    imdb = setfield(imdb, fields{fidx}, getfield(imdbs{1}, fields{fidx}));
  end
  for i = 1:n 
    for imdb_idx = 2:numel(imdb_paths)
      imdb.proposals{i} = [imdb.proposals{i} ; imdbs{imdb_idx}.proposals{i}];
      imdb.root_ids{i} = [imdb.root_ids{i} ; imdbs{imdb_idx}.root_ids{i}];
      if isfield(imdb, 'root_feat_types')
        imdb.root_feat_types{i} = [imdb.root_feat_types{i} ; imdbs{imdb_idx}.root_feat_types{i}];
      end
      if isfield(imdb, 'root_feat_types_code')
        imdb.root_feat_types_code{i} = [imdb.root_feat_types_code{i} ; imdbs{imdb_idx}.root_feat_types_code{i}];
      end
    end
    [imdb.proposals{i}, unique_ids] = unique(imdb.proposals{i}, 'rows');
    imdb.root_ids{i} = imdb.root_ids{i}(unique_ids);
    if isfield(imdb, 'root_feat_types')
      imdb.root_feat_types{i} = imdb.root_feat_types{i}(unique_ids);
    end
    if isfield(imdb, 'root_feat_types_code')
      imdb.root_feat_types_code{i} = imdb.root_feat_types_code{i}(unique_ids);
    end
  end
  savefile(save_path, imdb);
end
