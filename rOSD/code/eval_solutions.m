function [sol, corloc, detRate] = eval_solutions(args, class_indices)
% EVAL_SOLUTIONS
% [sol, corloc, detRate] = eval_solutions(args, class_indices)
%
% Perform post-processing on solutions and compute performance for each class.
%
% Parameters:
%
%   args: struct, each field is a cell array, which are
%
%     root: string, root folder of the image set.
%
%     clname: string, name of the class.
%
%     small_imdb: bool. 
%
%     lite_imdb: bool.
%
%     nu: int, maximum number of retained proposals in each image.
%
%     tau: int, maximum number of neighbors of each image.
%
%     nms_IoU: double, threshold in non-maximum suppression.
%
%     optim_sol: string, one of the initialization modes in 
%                {'one', 'infeasible', 'feasible'}.
%
%     save_result: bool.
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
%     iters: integer array.
%
%     ens_size: int, ensemble size.
%
%     num_nb_regions: int, number of regions in neighboring images 
%                        used to compute the score of each region.
%
%     K solutions will be divided into groups of args.ens_size solutions
%     for evaluation, args.iters is a subset of [1:round(K/args.ens_size)].
%
%   ---------------------
%
%   class_indices: array, indices of classes to compute solutions.
%
% Returns:
%
%

args = argument_reader(args);
corloc_nu = [];
corloc_5 = [];
corloc_1 = [];
detRate_nu = [];
detRate_5 = [];
detRate_1 = [];
for cl = class_indices
  opts = get_opts(args, cl);
  [msg, err] = argument_checker(opts);
  fprintf('%s\n', msg);
  if err 
    return;
  end
  opts.cl = cl;
  %------------------------------
  % get paths
  fprintf('Post processing for class %s\n', opts.clname);
  [opts.score_path, opts.sol_path, opts.save_path] = get_paths(opts);
  fprintf('Score path: %s\nSolution path: %s\nSave path: %s\n', ...
          opts.score_path, opts.sol_path, opts.save_path);
  %------------------------------
  fprintf('Loading imdb and reading class info ...\n');
  [opts.proposals, opts.bboxes, opts.group_ids, opts.imdb_path] = load_imdb(opts);
  opts.n = numel(opts.proposals, 1);
  opts.num_regions = cellfun(@(x) size(x,1), opts.proposals);
  %----------------------------------------------------
  fprintf('Creating data loaders and loading scores ...\n');
  opts.DL = opts.data_loader(load(opts.score_path));
  %----------------------------------------------------
  sol = struct('x_opt_nu', {[]}, 'x_opt_5', {[]}, 'x_opt_1', {[]}, 'x', {[]}, 'e_opt_1', {[]});
  corloc_nu_class = [];
  corloc_5_class = [];
  corloc_1_class = [];
  detRate_nu_class = [];
  detRate_5_class = [];
  detRate_1_class = [];
  opts.sol_buffer = get_first_solution_batch(opts);
  for iter = opts.iters
    fprintf('Iterations %d: ', iter);
    l = (iter-1)*opts.ens_size+1;
    r = iter*opts.ens_size;
    [opts.x, opts.e, opts.sol_buffer] = get_solutions(opts, opts.sol_buffer, l, r);
    %------------------------------
    % ENSEMBLE METHOD
    [sol.x_opt_nu{iter}, sol.x_opt_5{iter}, ...
     sol.x_opt_1{iter}, sol.e_opt_1{iter}] = ensemble_solutions(opts);
    corloc_nu_class(end+1) = CorLoc(opts.proposals, opts.bboxes, sol.x_opt_nu{iter}, 0.5);
    corloc_5_class(end+1) = CorLoc(opts.proposals, opts.bboxes, sol.x_opt_5{iter}, 0.5);
    corloc_1_class(end+1) = CorLoc(opts.proposals, opts.bboxes, sol.x_opt_1{iter}, 0.5);
    
    detRate_nu_class(end+1) = detection_rate(opts.proposals, opts.bboxes, sol.x_opt_nu{iter}, 0.5);
    detRate_5_class(end+1) = detection_rate(opts.proposals, opts.bboxes, sol.x_opt_5{iter}, 0.5);
    detRate_1_class(end+1) = detection_rate(opts.proposals, opts.bboxes, sol.x_opt_1{iter}, 0.5);
  
    sol.x{iter} = opts.x;
    fprintf('corloc_nu: %.2f, corloc_5: %.2f, corloc_1: %.2f\n', ...
            100*[corloc_nu_class(end), corloc_5_class(end), corloc_1_class(end)]);
    fprintf('detRate_nu: %.2f, detRate_5: %.2f, detRate_1: %.2f\n', ...
            100*[detRate_nu_class(end), detRate_5_class(end), detRate_1_class(end)]);
  end
  fprintf('Average: corloc_nu: %.2f / %.2f ; corloc_5: %.2f / %.2f ; corloc_1: %.2f / %.2f\n', ...
          100*mean(corloc_nu_class), 100* std(corloc_nu_class), ...
          100*mean(corloc_5_class), 100*std(corloc_5_class), ...
          100*mean(corloc_1_class), 100*std(corloc_1_class));
  corloc_nu = [corloc_nu; corloc_nu_class];
  corloc_5 = [corloc_5; corloc_5_class];
  corloc_1 = [corloc_1; corloc_1_class];
  detRate_nu = [detRate_nu; detRate_nu_class];
  detRate_5 = [detRate_5; detRate_5_class];
  detRate_1 = [detRate_1; detRate_1_class];
  % save results
  if opts.save_result
    corloc = struct('corloc_nu', corloc_nu_class, ...
                    'corloc_5', corloc_5_class, ...
                    'corloc_1', corloc_1_class);
    detRate = struct('detRate_nu', detRate_nu_class, ...
                     'detRate_5', detRate_5_class, ...
                     'detRate_1', detRate_1_class);
    save_result(opts, corloc, detRate, sol);
  end
end
corloc = struct('corloc_nu', corloc_nu, 'corloc_5', corloc_5, 'corloc_1', corloc_1);
detRate = struct('detRate_nu', detRate_nu, 'detRate_5', detRate_5, 'detRate_1', detRate_1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [args] = argument_reader(args)
  assert(all(isfield(args, {...
                'root', 'clname', 'small_imdb', 'lite_imdb', 'nu', 'tau', 'nms_IoU', ...
                'optim_sol', 'save_result', ... 
                'sctype', 'scname', 'normsc', 'rounding', 'iterating', 'data_loader', ...
                'prefiltered_nb', 'num_nb_rounding', 'num_nb_iterating', 'nb_root', 'nb_type' ...
                'iters', 'ens_size', 'num_nb_regions', ...
            })));
  num_classes = numel(args.root);
  for i = 1:num_classes
    if args.normsc{i}
      args.sol_home{i} = sprintf('solutions/norm_%d', args.nu{i});
      args.corloc_home{i} = sprintf('corloc/norm_%d', args.nu{i});
    else 
      args.sol_home{i} = sprintf('solutions/unnorm_%d', args.nu{i});
      args.corloc_home{i} = sprintf('corloc/unnorm_%d', args.nu{i});
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

function [proposals, bboxes, group_ids, imdb_path] = load_imdb(opts)
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
      group_code = arrayfun(@(j) sprintf('%s_%d', imdb.root_feat_types{i}{j}, ...
                                                  imdb.root_ids{i}(j)), ...
                            1:numel(imdb.root_ids{i}), 'Uni', false);
      [~, group_ids{i}] = ismember(group_code, unique(group_code));
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [score_path, sol_path, save_path] = get_paths(opts)
  % score path
  if opts.normsc
    score_path = fullfile(opts.root, opts.clname, opts.sctype, ...
                          sprintf('%s_normalized_%s.mat', opts.scname, opts.iterating));
  else
    score_path = fullfile(opts.root, opts.clname, opts.sctype, ...
                          sprintf('%s_%s.mat', opts.scname, opts.iterating)); 
  end
  % solution path
  if opts.prefiltered_nb
    sol_path = fullfile(opts.root, opts.sol_home, opts.clname, opts.optim_sol, ...
                        sprintf('%s/%s_%s_%s_%s', opts.nb_text, opts.scname, ...
                                                  opts.rounding, opts.iterating, ...
                                                  opts.sctype));
    save_path = fullfile(opts.root, opts.clname, opts.corloc_home, opts.optim_sol, ...
                        sprintf('%s/%s_%s_%s_%s', opts.nb_text, opts.scname, ...
                                                  opts.rounding, opts.iterating, ...
                                                  opts.sctype));
  else
    sol_path = fullfile(opts.root, opts.sol_home, opts.clname, opts.optim_sol, ...
                        sprintf('%s_%s_%s_%s', opts.scname, opts.rounding, ...
                                               opts.iterating, opts.sctype));
    save_path = fullfile(opts.root, opts.clname, opts.corloc_home, opts.optim_sol, ...
                        sprintf('%s_%s_%s_%s', opts.scname, opts.rounding, ...
                                               opts.iterating, opts.sctype));
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sorted_files] = sort_files(files)
  % All files are named as 'm_n.mat' where m and n are integers.
  begin_points = [];
  end_points = [];
  for i = 1:numel(files)
    filename = files{i};
    indices = cellfun(@str2num, strsplit(filename(1:end-4), '_'));
    begin_points = [begin_points indices(1)];
    end_points = [end_points indices(2)]; 
  end
  assert(numel(unique(begin_points)) == numel(begin_points));
  assert(numel(unique(end_points)) == numel(end_points));
  [~, min_idx] = sort(begin_points, 'ascend');
  sorted_files = files(min_idx);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [files] = get_solution_files(opts)
  files = dir(fullfile(opts.sol_path, '*_*.mat'));
  files = {files.name};
  files = sort_files(files);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sol] = get_first_solution_batch(opts)
  files = get_solution_files(opts);
  sol = load(fullfile(opts.sol_path, files{1}));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, e, sol_buffer] = get_solutions(opts, sol_buffer, l, r)
  % get solutions from files in sol_path and update 'sol_buffer'.
  % The funtion loads the needed file(s) such that the solutions
  % indexed from 'l' to 'r' are in 'sol_buffer' at the end.
  files = get_solution_files(opts);
  if numel(sol_buffer.x_s) < r | isempty(sol_buffer.x_s{l}) | isempty(sol_buffer.x_s{r})
    for i = 1:numel(files)
      ids = cellfun(@str2num, strsplit(files{i}(1:end-4), '_'));
      if numel(intersect([l:r], [ids(1):ids(2)])) > 0
        current_sol = load(fullfile(opts.sol_path, files{i}));
        sol_buffer.x_s(ids(1):ids(2)) = current_sol.x_s(ids(1):ids(2));
        sol_buffer.e_s(ids(1):ids(2)) = current_sol.e_s(ids(1):ids(2));
      end
    end
  end
  x_list = horzcat(sol_buffer.x_s{l:r});
  e_list = horzcat(sol_buffer.e_s{l:r});
  sol_buffer.x_s(l:r) = cell(1,r-l+1);
  sol_buffer.e_s(l:r) = cell(1,r-l+1);
  % combine solutions
  n = size(x_list, 1);
  x = cell(n,1);
  e = cell(n,1);
  for i =1:n
    x{i} = unique(cell2mat(x_list(i,:)));
    e{i} = unique(cell2mat(e_list(i,:)));
  end  
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_opt_nu, x_opt_5, x_opt_1, e_opt_1] = ensemble_solutions(opts)
  n = opts.n; nu = opts.nu; tau = opts.tau; nms_IoU = opts.nms_IoU;
  x_opt_nu = post_proc_nms(opts.DL, opts.x, opts.e, nu, opts.proposals, tau, 1, ...
                           opts.group_ids, nms_IoU);
  x_opt_5 = post_proc_nms(opts.DL, opts.x, opts.e, 5, opts.proposals, tau, 1, ...
                          opts.group_ids, nms_IoU);
  [x_opt_1, e_opt_1] = post_proc_nms(opts.DL, opts.x, opts.e, 1, opts.proposals, tau, ...
                                     opts.num_nb_regions, opts.group_ids, nms_IoU);
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_result(opts, corloc, detRate, sol)
  fprintf('Saving result ... ');
  mkdir(opts.save_path);
  save(fullfile(opts.save_path, ...
       sprintf('corloc_nms_IoU_0%s_ensemble_%d_iterations_%d_to_%d_nnbfor_%d.mat', ...
       num2str(10*opts.nms_IoU), opts.ens_size, opts.iters(1), opts.iters(end), opts.num_nb_regions)), ...
       '-struct', 'corloc');
  save(fullfile(opts.save_path, ...
       sprintf('detection_rate_nms_IoU_0%s_ensemble_%d_iterations_%d_to_%d_nnbfor_%d.mat', ...
       num2str(10*opts.nms_IoU), opts.ens_size, opts.iters(1), opts.iters(end), opts.num_nb_regions)), ...
       '-struct', 'detRate');
  save(fullfile(opts.save_path, ...
       sprintf('x_opt_nmsIoU_0%s_ensemble_%d_iterations_%d_to_%d_nnbfor_%d.mat', ...
       num2str(10*opts.nms_IoU), opts.ens_size, opts.iters(1), opts.iters(end), opts.num_nb_regions)), ...
       '-struct', 'sol');
  fprintf('DONE!\n');
end