root = fullfile(DATA_ROOT, imgset);

for cl = numel(classes)
  clname = classes{cl};
  proposal_path = fullfile(root, clname, [clname, '_', proposal_name]);
  files = dir(fullfile(proposal_path, '*_*.mat'));
  files = {files.name};
  num_files = numel(files);

  %--------------------------------
  % COMPRESS SIMILARITY
  imdb = struct;
  for i = 1:num_files
    filename = files{i};
    end_points = cellfun(@str2num, ...
                         strsplit(filename(1:end-4), '_'));
    rows = end_points(1):end_points(2);
    data = load(fullfile(proposal_path, filename));
    if isfield(data, 'root_feat_types')
      root_feat_types_code = {};
      for row = rows 
        root_feat_types_code{row,1} = 53*ones(numel(data.root_feat_types{row}),1);
      end
      data = rmfield(data, 'root_feat_types');
      data.root_feat_types_code = root_feat_types_code;
    end
    fields = fieldnames(data);
    for fidx = 1:numel(fields)
      imdb = setfield(imdb, fields{fidx}, {rows, 1}, getfield(data, fields{fidx}, {rows}));
    end
  end
  fprintf('\n');  
  save_path = fullfile(root, clname, [clname, '_', proposal_name, '.mat']);
  savefile(save_path, imdb);
end 
