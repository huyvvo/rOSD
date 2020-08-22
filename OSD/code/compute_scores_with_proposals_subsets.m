function [] = compute_scores(args, class_indices, row_indices)
% COMPUTE_SCORES
% [] = compute_scores(args)
%
% Function to compute confidence/standout scores.
%
% Parameters:
%
%   args.imgset: string, name of the image set.
%
%   args.small_imdb: bool, 
%
%   args.saveconf: bool, whether to save confidence scores.
%
%   args.savestd: bool, whether to save standout scores.
%
%   ---------------------
%
%   args.PHM_type: string, type of PHM function.
%
%   args.symmetric: bool, whether to symmetrize score matrices.
%
%   args.num_pos: int, number of positive entries in score matrices
%                         after sparsification.
%
%   args.num_pos_text: string, text code for num_pos.
%
%   args.stdver: int, version of the standout function.
%
%   args.max_iter: int, max_iteration in computing standout.
%
%   args.area_ratio: double, area ratio in the definition of background and 
%                    context regions.
%
%   args.area_ratio_text: string, text code for area_ratio.
%
%   ---------------------
%
%   args.deepfeat: bool, whether to use CNN features in the score computation.
%
%   args.feat_type: string, name of the features.
%
%   args.sim_type: string, text code for similarity function.
%
%   ---------------------
%
%   args.prefiltered_nb: bool, whether to only compute scores between an image
%                        and its neighbor candidates.
%
%   args.num_nb: int, number of neighbors of each image.
%
%   args.nb_root: string, root folder containing image pre-neighbors.
%
%   args.nb_type: string, type of the neighbors. 
%
% Returns: 
%
%

args = argument_reader(args);
[msg, err] = argument_checker(args);
args
fprintf('%s\n', msg);
if err 
  return;
end

for cl = class_indices
  opts.cl = cl;
  opts.clname = args.classes{cl};
  fprintf('Processing for class %s\n', opts.clname);
  %-------------------------------
  % load imdb
  [opts.images_size, opts.proposals, ~] = load_imdb(args, opts);
  opts.subproposals = load_subproposals(args, opts);
  opts.n = size(opts.proposals, 1);
  opts.indices = build_indices(args, opts);
  opts.feat_path = fullfile(args.feat_root, opts.clname, 'features/proposals', args.feat_type);
  [opts.save_path_conf, opts.save_path_std] = get_save_paths(args, opts);
  [opts.save_path_conf_sub, opts.save_path_std_sub] = get_save_paths_subproposals(args, opts);    
  %-------------------------------
  % compute scores
  S_confidence = {};
  S_standout = {};
  S_confidence_sub = {};
  S_standout_sub = {};
  for row = row_indices
    time_zero = tic;
    i = opts.indices(row, 1); j = opts.indices(row, 2);
    [confidence, standout, confidence_sub, standout_sub] = compute_scores_(args, opts, i, j);
    if args.saveconf
      S_confidence{row} = confidence;
      for sub_idx = 1:numel(args.root_subproposals)
        S_confidence_sub{sub_idx}{row} = confidence_sub{sub_idx};
      end
    end
    if args.savestd
      S_standout{row} = standout;
      for sub_idx = 1:numel(args.root_subproposals)
        S_standout_sub{sub_idx}{row} = standout_sub{sub_idx};
      end
    end
    fprintf('Scores computed in %.4f secs..........................\n', toc(time_zero));
  end
  % save result
  fprintf('Results will be saved to: \n%s\n%s\n', opts.save_path_conf, opts.save_path_std);
  save_results(args, opts, min(row_indices), max(row_indices), S_confidence, S_standout, S_confidence_sub, S_standout_sub);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {...
                        'imgset', 'imgset_subproposals', 'small_imdb', 'saveconf', 'savestd', 'save_big_scores', ...
                        'PHM_type', 'symmetric', 'num_pos', 'num_pos_text', ...
                        'stdver', 'max_iter', 'area_ratio', 'area_ratio_text', ...
                        'deepfeat', 'feat_type', 'sim_type', ...
                        'prefiltered_nb', 'nb_root', 'nb_type', 'num_nb', ...
                            })));
  args.root = ['~/', args.imgset];
  args.root_subproposals = fullfile('~/', args.imgset_subproposals);
  args.classes = get_classes(args.imgset);
  % get PHM function
  if strcmp(args.PHM_type, '')
    args.PHM_func = @PHM_lite;
  elseif strcmp(args.PHM_type, 'max')
    args.PHM_func = @PHM_lite_max;
  elseif strcmp(args.PHM_type, 'sum')
    args.PHM_func = @PHM_lite_sum;
  elseif strcmp(args.PHM_type, 'tmp')
    args.PHM_func = @PHM_lite_tmp;
  else 
    error('Unknown PHM function!');
  end

  % get stadout function or remove relating parameters if not necessary
  if args.savestd
    assert(isfield(args, 'stdver'))
    if args.stdver == 4
      args.stdfunc = @standout_box_pair_v4;
    else 
      error('Version of standout function not supported!');
    end
  end

  % get score name
  if args.deepfeat
    args.scname = [args.feat_type, '_', args.sim_type];
  else 
    args.scname = args.feat_type;
  end

  if args.symmetric
    args.scname = [args.scname, '_symmetric'];
  end 

  if ~strcmp(args.PHM_type, '') 
    args.scname = [args.scname, '_', args.PHM_type];
  end

  if args.prefiltered_nb
    assert(numel(args.num_nb) == numel(args.nb_type));
    assert(numel(args.num_nb) == numel(args.nb_root));
    for neighbor_idx = 1:numel(args.num_nb)
      args.scname = sprintf('%s_%d_%s', args.scname, args.num_nb, args.nb_type{neighbor_idx});
    end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [msg, err] = argument_checker(args)
  msg = 'Message from checker: ';
  err = false;
  if (args.small_imdb & (numel(args.feat_type) < 6 | ~strcmp(args.feat_type(1:6), 'small_'))) | ...
     (~args.small_imdb & (numel(args.feat_type) >= 6 & strcmp(args.feat_type(1:6), 'small_')))
      msg = sprintf('%s\n\t%s', msg, 'feat_type and small_imdb are not compatible');
      err = true;
  end

  if (args.num_pos == Inf & ~strcmp(args.num_pos_text, 'full')) | ...
     (args.num_pos < Inf & args.num_pos ~= str2double(args.num_pos_text))
    msg = sprintf('%s\n\t%s', msg, 'num_pos and num_pos_text are not compatible');
    err = true;
  end

  if args.deepfeat & ~any(cellfun(@(x) strcmp(x, args.sim_type) == 1, ...
                                  {'sp', 'cos', 'spatial_cos', '01', 'spatial_01', 'sqrt', 'log'}))
    msg = sprintf('%s\n\t%s', msg, 'sim_type must be in {"sp", "cos", "01", "sqrt", "log"}');
    err = true;
  end

  msg = sprintf('%s\nEnd message.\n', msg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [images_size, proposals, bboxes] = load_imdb(args, opts)
  if args.small_imdb 
    imdb = load(fullfile(args.root, opts.clname, [opts.clname, '_small_lite.mat']));
  else 
    imdb = load(fullfile(args.root, opts.clname, [opts.clname, '_lite.mat']));
  end
  images_size = imdb.images_size;
  proposals = imdb.proposals;
  bboxes = imdb.bboxes;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [subproposals] = load_subproposals(args, opts)
  subproposals = cell(numel(args.root_subproposals),1);
  for sub_idx = 1:numel(subproposals)
    if args.small_imdb 
      imdb = load(fullfile(args.root_subproposals{sub_idx}, opts.clname, [opts.clname, '_small_lite.mat']));
    else 
      imdb = load(fullfile(args.root_subproposals{sub_idx}, opts.clname, [opts.clname, '_lite.mat']));
    end
    subproposals{sub_idx} = imdb.proposal_indices;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e] = load_neighbors_single_(args, opts, idx)
  nb_path = fullfile(args.nb_root{idx}, args.nb_type{idx}, opts.clname, ...
                     sprintf('%d.mat', args.num_nb));
  e = getfield(load(nb_path), 'e');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e] = load_neighbors_multiple(args, opts)
  ids = 1:numel(args.nb_root);
  e = load_neighbors_single_(args, opts, ids(1));
  for idx = ids(2:end)
    current_e = load_neighbors_single_(args, opts, idx);
    e = arrayfun(@(i) [e{i}, current_e{i}], [1:size(current_e,1)]', 'Uni', false);
  end 
  e = cellfun(@(x) unique(x), e, 'Uni', false);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [indices] = build_indices(args, opts)
  n = opts.n; 
  if args.prefiltered_nb
    e = load_neighbors_multiple(args, opts);
    assert(size(e,1) == n, sprintf('Size of e is %d\n', size(e)));
    actual_num_nb = cellfun(@numel, e);
    indices = [repelem([1:n]', actual_num_nb) reshape(cell2mat(e'), [], 1)];
    indices = [min(indices'); max(indices')]';
    indices = unique(indices, 'rows');
  else 
    indices = [repelem([1:n]',n,1) repmat([1:n]',n,1)];
    indices = indices(indices(:,1) < indices(:,2), :);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [save_path_conf, save_path_std] = get_save_paths(args, opts)
  save_path_conf = fullfile(args.root, opts.clname, 'confidence', ...
                            sprintf('%s_normalized_%s', args.scname, args.num_pos_text));
  save_path_std = fullfile(args.root, opts.clname, 'standout', ...
                           sprintf('%s_v%d_%s_normalized_%s', ...
                                   args.scname, args.stdver, ...
                                   args.area_ratio_text, args.num_pos_text));  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [save_path_conf, save_path_std] = get_save_paths_subproposals(args, opts)
  save_path_conf = cell(numel(args.root_subproposals),1);
  save_path_std = cell(numel(args.root_subproposals),1);
  for sub_idx = 1:numel(args.root_subproposals)
    save_path_conf{sub_idx} = fullfile(args.root_subproposals{sub_idx}, opts.clname, 'confidence', ...
                              sprintf('%s_subproposals_normalized_%s', args.scname, args.num_pos_text));
    save_path_std{sub_idx} = fullfile(args.root_subproposals{sub_idx}, opts.clname, 'standout', ...
                             sprintf('%s_v%d_%s_subproposals_normalized_%s', ...
                                     args.scname, args.stdver, ...
                                     args.area_ratio_text, args.num_pos_text));  
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [feati, featj] = read_feat(args, opts, i, j)
  feati = getfield(load(fullfile(opts.feat_path, sprintf('%d.mat', i))), 'data');
  featj = getfield(load(fullfile(opts.feat_path, sprintf('%d.mat', j))), 'data');
  if args.deepfeat
    feati = process_deepfeat(feati, args.sim_type);
    featj = process_deepfeat(featj, args.sim_type);
  end
  assert(numel(size(feati)) == 2);
  assert(numel(size(featj)) == 2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [confidence, standout, confidence_sub, standout_sub] = compute_scores_(args, opts, i, j)
  [feati, featj] = read_feat(args, opts, i, j);
  % compute standout
  timer_conf = tic;
  if args.symmetric
    score1 = PHM_confidence_lite(args.PHM_func, ...
                                 opts.images_size{i}, opts.images_size{j}, ...
                                 opts.proposals{i}, opts.proposals{j}, ...
                                 feati, featj, 'RAW');
    score2 = PHM_confidence_lite(args.PHM_func, ...
                                 opts.images_size{j}, opts.images_size{i}, ...
                                 opts.proposals{j}, opts.proposals{i}, ...
                                 featj, feati, 'RAW');
    confidence = max(score1, 0) + max(score2', 0);
  else 
    score = PHM_confidence_lite(args.PHM_func, ...
                                opts.images_size{i}, opts.images_size{j}, ...
                                opts.proposals{i}, opts.proposals{j}, ...
                                feati, featj, 'RAW');
    confidence = max(score, 0);
  end
  confidence = confidence/prod(size(confidence))*1e6;
  fprintf('Confidence computed in %f sec\n', toc(timer_conf));
  % compute and sparsify standout
  timer_std = tic;
  if args.savestd
    standout = args.stdfunc(transpose(opts.proposals{i}), transpose(opts.proposals{j}), ...
                            confidence, args.max_iter, args.area_ratio);
    if args.num_pos ~= Inf
      for sub_idx = 1:numel(args.root_subproposals)
        standout_sub{sub_idx} = sparsify_matrix(standout(opts.subproposals{sub_idx}{i}, opts.subproposals{sub_idx}{j}), args.num_pos);
      end
      if args.save_big_scores
        standout = sparsify_matrix(standout, args.num_pos);
      else
        standout = [];
      end
    else
      for sub_idx = 1:numel(args.root_subproposals) 
        standout_sub{sub_idx} = standout(opts.subproposals{sub_idx}{i}, opts.subproposals{sub_idx}{j});
      end
      if args.save_big_scores
        standout = sparse(double(standout));
      else 
        standout = [];
      end
    end
  else 
    standout = [];
    standout_sub = [];
  end
  fprintf('Standout computed in %f secs\n', toc(timer_std));
  % sparsify confidence
  if args.saveconf
    if args.num_pos ~= Inf
      for sub_idx = 1:numel(args.root_subproposals)
        confidence_sub{sub_idx} = sparsify_matrix(confidence(opts.subproposals{sub_idx}{i}, opts.subproposals{sub_idx}{j}), args.num_pos);
      end
      if args.save_big_scores
        confidence = sparsify_matrix(confidence, args.num_pos);
      else 
        confidence = [];
      end
    else 
      for sub_idx = 1:numel(args.root_subproposals)
        confidence_sub = confidence(opts.subproposals{sub_idx}{i}, opts.subproposals{sub_idx}{j});
      end
      if args.save_big_scores
        confidence = sparse(double(confidence));
      else 
        confidence = [];
      end
    end
  else 
    confidence = [];
    confidence_sub = [];
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_results(args, opts, left, right, ...
                           S_confidence, S_standout, S_confidence_sub, S_standout_sub)
  if args.saveconf
    fprintf('Saving confidence score ...\n');
    mkdir(opts.save_path_conf);
    if exist(fullfile(opts.save_path_conf, 'indices.mat'), 'file') ~= 2
      indices = opts.indices;
      save(fullfile(opts.save_path_conf, 'indices.mat'), 'indices');
    else 
      fprintf('Indices.mat in confidence save path has been created!\n');
    end
    if args.save_big_scores
      data.data = S_confidence(left:right);
      savefile(fullfile(opts.save_path_conf, sprintf('%d_%d.mat', left, right)), data);
    end 

    for sub_idx = 1:numel(args.root_subproposals)
      mkdir(opts.save_path_conf_sub{sub_idx});
      if exist(fullfile(opts.save_path_conf_sub{sub_idx}, 'indices.mat'), 'file') ~= 2
        indices = opts.indices;
        save(fullfile(opts.save_path_conf_sub{sub_idx}, 'indices.mat'), 'indices');
      else 
        fprintf('Indices.mat in confidence save path has been created!\n');
      end
      data.data = S_confidence_sub{sub_idx}(left:right);
      savefile(fullfile(opts.save_path_conf_sub{sub_idx}, sprintf('%d_%d.mat', left, right)), data);
    end
  end

  if args.savestd
    fprintf('Saving standout score ...\n');
    mkdir(opts.save_path_std);
    if exist(fullfile(opts.save_path_std, 'indices.mat'), 'file') ~= 2
      save(fullfile(opts.save_path_std, 'indices.mat'), 'indices');
    else 
      fprintf('Indices.mat in standout save path has been created!\n');
    end
     if args.save_big_scores
      data.data = S_standout(left:right);
      savefile(fullfile(opts.save_path_std, sprintf('%d_%d.mat', left, right)), data);
    end 

    for sub_idx = 1:numel(args.root_subproposals)
      mkdir(opts.save_path_std_sub{sub_idx});
      if exist(fullfile(opts.save_path_std_sub{sub_idx}, 'indices.mat'), 'file') ~= 2
        save(fullfile(opts.save_path_std_sub{sub_idx}, 'indices.mat'), 'indices');
      else 
        fprintf('Indices.mat in standout save path has been created!\n');
      end
      data.data = S_standout_sub{sub_idx}(left:right);
      savefile(fullfile(opts.save_path_std_sub{sub_idx}, sprintf('%d_%d.mat', left, right)), data);
    end
  end
end
