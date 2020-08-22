root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);
%-------------------------------
% SET PARAMETERS
% 'confidence' or 'standout' 
score_type = 'confidence';
[score_name{all_classes}] = deal('vgg19_relu54_matconvnet_77_roi_pooling_noresize_01_symmetric_normalized_1000');
[score_name_to_save{all_classes}] = deal('vgg19_relu54_matconvnet_77_roi_pooling_noresize_01_symmetric_normalized_1000.mat');
[score_name{big_classes}] = deal('vgg19_relu54_matconvnet_77_roi_pooling_noresize_01_symmetric_50_neighbor_cos_vgg19_fc6_matconvnet_resize_normalized_1000');
[score_name_to_save{big_classes}] = deal('vgg19_relu54_matconvnet_77_roi_pooling_noresize_01_symmetric_50_neighbor_cos_vgg19_fc6_matconvnet_resize_normalized_1000.mat');


for cl = 1:numel(classes)
  clname = classes{cl};
  fprintf('Compress %s for class %s\n', score_type, clname);
  imdb = load(fullfile(root, clname, [clname, '_lite.mat']), 'bboxes');
  n = size(imdb.bboxes, 1);
  % get score files
  score_path = fullfile(root, clname, score_type, score_name{cl});
  files = dir(fullfile(score_path, '*_*.mat')); 
  files = {files.name};
  num_files = numel(files);
  % indices.mat contains corresponding subscripts of score matrices 
  indices = getfield(load(fullfile(score_path, 'indices.mat')), 'indices');

  S = cell(n,n);
  for i = 1:num_files
    % read indices from filenames
    filename = files{i};
    end_points = cellfun(@str2num, strsplit(filename(1:end-4), '_'));
    rows = end_points(1):end_points(2);
    fprintf('Processing for rows %d to %d\n', end_points);
    % load scores
    scores = getfield(load(fullfile(score_path, filename)), 'data');
    S(sub2ind([n,n], indices(rows,1), indices(rows,2))) = scores(rows-end_points(1)+1);
  end
  fprintf('\n');
  % save result
  mkdir(fullfile(root, clname, [score_type, '_imdb']));
  data = struct;
  data.S = S;
  savefile(fullfile(root, clname, [score_type, '_imdb'], score_name_to_save{cl}), data);
end
