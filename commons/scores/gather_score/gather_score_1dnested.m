assert(exist('row_indices', 'var') == 1);

PWD = pwd();
cd ~/code/uod
add_path;
cd(PWD);

imgsets = {...
           'coco_train_20k_vgg19_matconvnet_cnn_03_20_1_05_with_roots', ...
           };

for imgset_idx = 1:numel(imgsets)
  args.imgset = imgsets{imgset_idx};
  root = ['~/', args.imgset];
  classes = get_classes(args.imgset);
  %-------------------------------
  % SET PARAMETERS
  % 'confidence' or 'standout' 
  args.score_type = 'confidence';
  args.score_name = 'vgg19_relu54_matconvnet_77_roi_pooling_noresize_01_symmetric_50_neighbor_cos_vgg19_fc6_matconvnet_resize_normalized_1000';
  %-------------------------------
  % number of top entries to keep in each score matrixs
  args.num_keep = 50;
  args.num_keep_text = '50';
  %-------------------------------
  args = argument_reader(args);
  fprintf('%s\n', argument_checker(args));
  args

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% MAIN CODE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  for cl = row_indices
    clname = classes{cl};
    fprintf('Compress %s for class %s\n', args.score_type, clname);
    fprintf('Score name is %s\n', args.score_type);

    % get number of images
    imdb = load(fullfile(root, clname, [clname, '_lite.mat']));
    n = size(imdb.proposals, 1);
    clear imdb;

    % get score files
    score_path = fullfile(root, clname, args.score_type, args.score_name);
    % get only score files, not indices.mat
    files = dir(fullfile(score_path, '*_*.mat')); 
    files = {files.name};
    num_files = numel(files);

    % indices.mat contains corresponding subscripts of score matrices 
    indices = getfield(load(fullfile(score_path, 'indices.mat')), 'indices');
    % get undirected neighbor lists of images
    adj = sparse(indices(:,1), indices(:,2), ones(size(indices,1),1), n, n);
    adj = max(adj, adj');
    fprintf('Building e, rows ');
    e = cell(n,1);
    for i = 1:n 
      if mod(i,100) == 1
        fprintf('%d ', i);
      end
      e{i} = find(adj(i,:));
    end
    clear adj;
    % get neighbor positions in S
    % S_{ij} = S{i}{neighbor_positions(i,j)} if
    % i < j and j is an UNDIRECTED potential neighbor of i.
    neighbor_positions = sparse(n,n);
    for i = 1:n
      % get neighbors of i that have score matrices stored in S{i}
      % make sure that indices of these neighbors are in increasing order
      valid_neighbors = e{i}(e{i} > i);
      if numel(valid_neighbors) > 1
        assert(all(valid_neighbors(2:end) > valid_neighbors(1:end-1)));
      end
      neighbor_positions(i, valid_neighbors) = 1:numel(valid_neighbors);
    end

    % begin collecting scores
    S = cell(n,1);
    for i = 1:n 
      S{i} = cell(0);
    end
    for file_idx = 1:num_files
      % read indices from filenames
      filename = files{file_idx};
      end_points = cellfun(@str2num, ...
                           strsplit(filename(1:end-4), '_'));
      rows = end_points(1):end_points(2);
      fprintf('Processing for rows %d to %d\n', end_points);
      % load scores
      scores = getfield(load(fullfile(score_path, filename)), 'data');
      % sparsify score matrices if necessary (num_keep < Inf)
      if args.num_keep ~= Inf
        for row = rows 
          scores{row-end_points(1)+1} = sparsify_matrix(scores{row-end_points(1)+1}, args.num_keep);
        end
      end
      for row = rows 
        i = indices(row,1); j = indices(row,2);
        S{i}{neighbor_positions(i,j)} = scores{row-end_points(1)+1};
      end
    end
    fprintf('\n');

    % save result
    save_path = fullfile(root, clname, [args.score_type, '_1dnestedcell_imdb']);
    mkdir(save_path);
    data.S = S;
    data.e = e;
    savefile(fullfile(save_path, args.score_name_to_save), data);
  end
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {'score_type', 'score_name', 'num_keep'})));
  pos = strfind(args.score_name, '_');
  pos = pos(end);
  args.score_name_to_save = sprintf('%s_%s.mat', args.score_name(1:pos-1), args.num_keep_text);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function msg = argument_checker(args)
  msg = 'Message from checker: ';
  if strcmp(args.score_type, 'confidence') == 0 && strcmp(args.score_type, 'standout') == 0
      msg = sprintf('%s\n\t%s', msg, 'score_type must be "confidence" or "standout"');
  end
  msg = sprintf('%s\nEnd message.\n', msg);
end

