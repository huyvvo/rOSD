function [corloc_1, corloc_5, corloc_nu, sol] = compute_solutions(args, class_indices, solution_indices)
% COMPUTE_SOLUTIONS
% [corloc_1, corloc_5, corloc_nu, sol] = compute_solutions(args, class_indices, solution_indices)
%
% Parameters:
%
%   args: struct, each field is a cell array, which are
%
%     root: string, root folder of the image set.
%
%     clname: string, name of the class.
%
%     small_imdb: bool, 
%
%     lite_imdb: bool,
%
%     nu: int, maximum number of retained proposals in each image.
%
%     tau: int, maximum number of neighbors of each image.
%
%     nms_IoU: double, threshold in non-maximum suppression.
%
%     ---------------------
%
%     sctype: string, score_type, examples 'confidence_imdb' or 
%                'standout_imdb'.
%
%     scname: string, name of the score.
%
%     normsc: bool, if the scores to use is normalized or not.
%
%     rounding: string, code for the number of non-negative elements
%               in the score matrices for rounding.
%
%     iterating: string, code for the number of non-negative elements
%                in the score matrices for block coordinate ascent.
%
%     data_loader: a DataLoader class. 
%
%     ---------------------
%
%     prefiltered_nb: bool, whether to use scores with pre-filtered 
%                     neighbors.
%
%     num_nb_rounding: int, number of neighbors of each image in score
%                      DataLoader used in rounding step.
%
%     num_nb_iterating: int, number of neighbors of each image in score
%                       DataLoader used in block coordinate ascent.
%
%     nb_root: string, root folder containing image pre-neighbors.
%
%     nb_type: string, type of the neighbors.
%
%     ---------------------
%
%     save_result: bool, whether to save results.
%
%     vis_result: bool. whether to visualize results.
%
%     vis_root: string, path to folder containing visualizations.
%
%     ---------------------
%
%     optim_sol: string, one of the initialization modes in 
%                {'one', 'infeasible', 'feasible'}.
%
%     res_path: string, path to the file containing continuous
%               optimization solution. It must be not empty when
%               optim_sol is 'infeasible' or 'feasible'.
%
%     ---------------------
%
%     verbose: bool, whether to print messages into console.
%
%   ---------------------
%
%   class_indices: array, indices of classes to compute solutions.
%
%   solution_indices: array, indices of solutions to compute.   
%
% Returns
%
%

args = argument_reader(args);
corloc_nu = [];
corloc_5 = [];
corloc_1 = [];
for cl = class_indices
  opts = get_opts(args, cl);
  [msg, err] = argument_checker(opts);
  fprintf('%s\n', msg);
  if err 
    return;
  end
  opts.cl = cl;
  %------------------------------
  % load imdb
  fprintf('Computing solutions for class %s\n', opts.clname);
  [opts.proposals, opts.bboxes, opts.imdb_path, opts.group_ids] = load_imdb(opts);
  opts.n = numel(opts.proposals);
  opts.num_regions = cellfun(@(x) size(x,1), opts.proposals);
  %------------------------------
  % initialization
  [opts.x, opts.e] = initialize_xe(opts);
  %------------------------------
  % load scores and create data loaders
  fprintf('Loading scores and creating data loaders\n');
  [scp_rounding, scp_iterating] = get_score_paths(opts);
  if strcmp(scp_rounding, scp_iterating) == 1
    opts.DL = opts.data_loader(load(scp_rounding));
  else 
    opts.rounding_DL = opts.data_loader(load(scp_rounding));
    opts.iterating_DL = opts.data_loader(load(scp_iterating));
  end
  %------------------------------
  % load neighbors
  [opts.e_rounding, opts.e_iterating] = load_neighbors(opts);
  %------------------------------
  % compute solutions
  corloc_nu_class = [];
  corloc_1_class = [];
  sol = struct('x_s', {[]}, 'e_s', {[]}, 'x_opt_5', {[]}, 'x_opt_1', {[]}, 'e_opt_1', {[]}, 'nms_IoU', opts.nms_IoU);
  for i = solution_indices
    if mod(i, 100) == 1
      fprintf('Computing solution %d\n', i);
    end
    tic;
    [sol.x_s{i}, sol.e_s{i}, sol.x_opt_5{i}, sol.x_opt_1{i}, sol.e_opt_1{i}] = get_solutions(opts);
    corloc_nu_class = [corloc_nu_class CorLoc(opts.proposals, opts.bboxes, sol.x_s{i}, 0.5)];
    corloc_5_class = [corloc_1_class CorLoc(opts.proposals, opts.bboxes, sol.x_opt_5{i}, 0.5)];
    corloc_1_class = [corloc_1_class CorLoc(opts.proposals, opts.bboxes, sol.x_opt_1{i}, 0.5)];
    fprintf('Solution computed in %.2f secs\n', toc);
    fprintf('CorLoc nu/5/1: %.4f/%.4f/%.4f\n', corloc_nu_class(end), corloc_5_class(end), corloc_1_class(end));
  end
  fprintf('\n');
  fprintf('CorLoc_1: %.2f +/- %.2f\n', 100*mean(corloc_1_class), 100*std(corloc_1_class));
  corloc_nu = [corloc_nu; corloc_nu_class];
  corloc_5 = [corloc_5; corloc_5_class];
  corloc_1 = [corloc_1; corloc_1_class];
  if opts.vis_result
    visualize_result(opts.imdb_path([1:end-9 end-3:end]), fullfile(opts.vis_root, opts.clname), ...
                     sol.x_s{solution_indices(1)}, sol.x_opt_5{solution_indices(1)}, sol.x_opt_1{solution_indices(1)});
  end
  %------------------------------
  % save results
  opts.save_path = get_save_path(opts);
  fprintf('Results will be saved to %s\n', opts.save_path);
  if opts.save_result
    save_result(opts, sol, solution_indices);
  end
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {...
                'root', 'clname', 'small_imdb', 'lite_imdb', 'nu', 'tau', 'nms_IoU', ...
                'sctype', 'scname', 'normsc', 'rounding', 'iterating', 'data_loader', ...
                'prefiltered_nb', 'num_nb_rounding', 'num_nb_iterating', 'nb_root', 'nb_type' ...
                'optim_sol', 'res_path', ...
                'save_result', 'vis_result', 'vis_root', 'verbose', ...
            })));
  num_classes = numel(args.root);
  for i = 1:num_classes
    if args.normsc{i}
      args.sol_home{i} = sprintf('solutions/norm_%d', args.nu{i});
    else 
      args.sol_home{i} = sprintf('solutions/unnorm_%d', args.nu{i});;
    end
    if args.prefiltered_nb{i}
      assert(numel(args.nb_type{i}) == numel(args.nb_root{i}));
      args.nb_text{i} = join(args.nb_type{i}, '_');
      args.nb_text{i} = args.nb_text{i}{1};
      args.nb_text{i} = sprintf('%s_%d_%d', args.nb_text{i}, ...
                                            args.num_nb_rounding{i}, ...
                                            args.num_nb_iterating{i});
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [opts] = get_opts(args, cl)
  opts = struct;
  fields = fieldnames(args);
  for i = 1:numel(fields)
    fieldvalues = getfield(args, fields{i});
    opts = setfield(opts, fields{i}, fieldvalues{cl});
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [msg, err] = argument_checker(opts)
  msg = 'Verifying arguments: ';
  err = false;
  if (opts.small_imdb & (numel(opts.scname) < 6 | ~strcmp(opts.scname(1:6), 'small_'))) | ...
     (~opts.small_imdb & (numel(opts.scname) >= 6 & strcmp(opts.scname(1:6), 'small_')))
      msg = sprintf('%s\n\t%s', msg, 'score_name and small_imdb are not compatible');
      err = true;
  end

  if ~any(cellfun(@(x) isequal(opts.optim_sol, x), {'one', 'infeasible', 'feasible'}))
    msg = sprintf('%s\n\t%s', msg, 'initialization must be in {"one", "infeasible", "feasible"}');
    err = true;
  end

  msg = sprintf('%s...End message.', msg);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [proposals, bboxes, imdb_path, group_ids] = load_imdb(opts)
  if opts.small_imdb
    if opts.lite_imdb
      imdb_path = fullfile(opts.root, opts.clname, [opts.clname, '_small_lite.mat']);
    else
      imdb_path = fullfile(opts.root, opts.clname, [opts.clname, '_small.mat']);
    end 
  else 
    if opts.lite_imdb
      imdb_path = fullfile(opts.root, opts.clname, [opts.clname, '_lite.mat']);
    else 
      imdb_path = fullfile(opts.root, opts.clname, [opts.clname, '.mat']);
    end  
  end
  imdb = load(imdb_path);
  proposals = imdb.proposals;
  bboxes = imdb.bboxes;
  if isfield(imdb, 'group_ids')
    group_ids = imdb.group_ids;
  else 
    n = numel(proposals);
    group_ids = cell(n,1);
    for i = 1:n
      if isfield(imdb, 'root_feat_types_code') 
        group_code = imdb.root_feat_types_code{i}*1e6 + imdb.root_ids{i};
        group_code = group_code(:)';
      else 
        group_code = arrayfun(@(j) sprintf('%s_%d', imdb.root_feat_types{i}{j}, ...
                                                    imdb.root_ids{i}(j)), ...
                              1:numel(imdb.root_ids{i}), 'Uni', false);
      end
      [~, group_ids{i}] = ismember(group_code, unique(group_code));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [scp_rounding, scp_iterating] = get_score_paths(opts)
  if opts.normsc
    scp_rounding = fullfile(opts.root, opts.clname, opts.sctype, ...
                            sprintf('%s_normalized_%s.mat', opts.scname, opts.rounding));
    scp_iterating = fullfile(opts.root, opts.clname, opts.sctype, ...
                             sprintf('%s_normalized_%s.mat', opts.scname, opts.iterating));
  else
    scp_rounding = fullfile(opts.root, opts.clname, opts.sctype, ...
                            sprintf('%s_%s.mat', opts.scname, opts.rounding));
    scp_iterating = fullfile(opts.root, opts.clname, opts.sctype, ...
                             sprintf('%s_%s.mat', opts.scname, opts.iterating)); 
  end
  if exist(scp_rounding, 'file') ~= 2
    error(sprintf('Rounding score file %s does not exist', scp_rounding));
  end
  if exist(scp_iterating, 'file') ~= 2
    error(sprintf('Iterating score file %s does not exist', scp_iterating));
  end 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e_rounding, e_iterating] = load_neighbors_single_(opts, idx)
  nbp_rounding = fullfile(opts.nb_root{idx}, opts.nb_type{idx}, opts.clname, ...
                          sprintf('%d.mat', opts.num_nb_rounding));
  e_rounding = getfield(load(nbp_rounding), 'e');

  nbp_iterating = fullfile(opts.nb_root{idx}, opts.nb_type{idx}, opts.clname, ...
                           sprintf('%d.mat', opts.num_nb_iterating));
  e_iterating = getfield(load(nbp_iterating), 'e');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [e_rounding, e_iterating] = load_neighbors(opts)
  if opts.prefiltered_nb
    ids = 1:numel(opts.nb_root);
    [e_rounding, e_iterating] = load_neighbors_single_(opts, ids(1));
    for idx = ids(2:end)
      [e_r, e_i] = load_neighbors_single_(opts, idx);
      e_rounding = arrayfun(@(i) [e_rounding{i}, e_r{i}], [1:size(e_r,1)]', 'Uni', false);
      e_iterating = arrayfun(@(i) [e_iterating{i}, e_i{i}], [1:size(e_i,1)]', 'Uni', false);
    end 
    e_rounding = cellfun(@(x) unique(x), e_rounding, 'Uni', false);
    e_iterating = cellfun(@(x) unique(x), e_iterating, 'Uni', false);
  else 
    e_rounding = arrayfun(@(i) setdiff(1:opts.n,i), [1:opts.n]', 'Uni', false);
    e_iterating = e_rounding;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [save_path] = get_save_path(opts)
  if opts.prefiltered_nb
    save_path = fullfile(opts.root, opts.sol_home, opts.clname, opts.optim_sol, ...
                         sprintf('%s/%s_%s_%s_%s', opts.nb_text, opts.scname, ...
                                                   opts.rounding, opts.iterating, ...
                                                   opts.sctype));
  else
    save_path = fullfile(opts.root, opts.sol_home, opts.clname, opts.optim_sol, ...
                         sprintf('%s_%s_%s_%s', opts.scname, opts.rounding, ...
                                                opts.iterating, opts.sctype));
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x,e] = initialize_xe(opts)
  n = opts.n;
  num_regions = opts.num_regions;
  if strcmp(opts.optim_sol, 'one') == 1
    x = mat2cell(ones(1, sum(num_regions)), [1], num_regions)';
    e = ones(n);
    e(sub2ind([n,n], 1:n, 1:n)) = 0;
  else
    xe = getfield(load(opts.res_path), 'xe_run');
    [x,e] = get_x_e(xe, n, num_regions);
    if strcmp(opts.optim_sol, 'feasible') == 1
      [x,e] = rescale_x_e_2(x, e, opts.nu, opts.tau);
    end
  end
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_s, e_s, x_opt_5, x_opt_1] = get_solutions(opts)
  n = opts.n; nu = opts.nu; tau = opts.tau; 
  if isfield(opts, 'DL') == 1
    [x_s, e_s] = round_x_e(opts.DL, opts.x, opts.e, opts.e_rounding, nu, tau, ...
                           opts.group_ids, randperm(n,n));
    [x_s, e_s] = ascent_x_e(opts.DL, x_s, e_s, opts.e_iterating, nu, tau, ...
                            opts.num_regions, opts.group_ids, 10);
    x_opt_5 = post_proc_nms(opts.DL, x_s, e_s, 5, opts.proposals, tau, 1, ...
                            opts.group_ids, opts.nms_IoU);
    [x_opt_1, e_opt_1] = post_proc_nms(opts.DL, x_s, e_s, 1, opts.proposals, tau, 1, ...
                                       opts.group_ids, opts.nms_IoU);
  else 
    [x_s, e_s] = round_x_e(opts.rounding_DL, opts.x, opts.e, opts.e_rounding, nu, tau, ...
                           opts.group_ids, randperm(n,n));
    [x_s, e_s] = ascent_x_e(opts.iterating_DL, x_s, e_s, opts.e_iterating, nu, tau, ...
                            opts.num_regions, opts.group_ids, 10);
    x_opt_5 = post_proc_nms(opts.iterating_DL, x_s, e_s, 5, opts.proposals, tau, 1, ...
                            opts.group_ids, opts.nms_IoU);
    [x_opt_1, e_opt_1] = post_proc_nms(opts.iterating_DL, x_s, e_s, 1, opts.proposals, tau, 1, ...
                                       opts.group_ids, opts.nms_IoU);
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_result(opts, solutions, indices)
  fprintf('Saving results ... ');
  mkdir(opts.save_path);
  save(fullfile(opts.save_path, sprintf('%d_%d.mat', indices(1), indices(end))), ...
       '-struct', 'solutions');
  fprintf('DONE!\n');
end 
