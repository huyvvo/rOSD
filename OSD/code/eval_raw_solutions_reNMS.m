function [corloc, detRate] = eval_raw_solutions(args, class_indices)
% EVAL_RAW_SOLUTIONS_RENMS
% [sol, corloc_nu, corloc_5, corloc_1] = eval_raw_solutions_reNMS(args, class_indices)
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
%     num_iters: integer array.
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
  [opts.proposals, opts.bboxes, opts.imdb_path] = load_imdb(opts);
  opts.n = numel(opts.proposals, 1);
  opts.num_regions = cellfun(@(x) size(x,1), opts.proposals);
  %----------------------------------------------------
  corloc = struct('corloc_nu', [], 'corloc_5', [], 'corloc_1', []);
  detRate = struct('detRate_nu', [], 'detRate_5', [], 'detRate_1', []);
  sol = struct('x_opt_nu', {[]}, 'x_opt_5', {[]}, 'x_opt_1', {[]}, 'e_opt_1', {[]});
  opts.sol_buffer = get_first_solution_batch(opts);
  opts.sol_buffer = get_solutions(opts, opts.sol_buffer, 1, opts.num_iters);
  %----------------------------------------------------
  fprintf('Creating data loaders and loading scores ...\n');
  opts.DL = opts.data_loader(load(opts.score_path));
  %----------------------------------------------------
  for iter = 1:opts.num_iters
    [x_opt_nu, x_opt_5, x_opt_1, e_opt_1] = process_solutions(opts, iter);

    sol.x_opt_nu{end+1} = x_opt_nu;
    sol.x_opt_5{end+1} = x_opt_5;
    sol.x_opt_1{end+1} = x_opt_1;
    sol.e_opt_1{end+1} = e_opt_1;

    corloc.corloc_nu(end+1) = CorLoc(opts.proposals, opts.bboxes, x_opt_nu, 0.5);
    corloc.corloc_5(end+1) = CorLoc(opts.proposals, opts.bboxes, x_opt_5, 0.5);
    corloc.corloc_1(end+1) = CorLoc(opts.proposals, opts.bboxes, x_opt_1, 0.5);
    
    detRate.detRate_nu(end+1) = detection_rate(opts.proposals, opts.bboxes, x_opt_nu, 0.5);
    detRate.detRate_5(end+1) = detection_rate(opts.proposals, opts.bboxes, x_opt_5, 0.5);
    detRate.detRate_1(end+1) = detection_rate(opts.proposals, opts.bboxes, x_opt_1, 0.5);
  end
  fprintf('Average: corloc_nu: %.2f / %.2f ; corloc_5: %.2f / %.2f ; corloc_1: %.2f / %.2f\n', ...
          100*mean(corloc.corloc_nu), 100* std(corloc.corloc_nu), ...
          100*mean(corloc.corloc_5), 100*std(corloc.corloc_5), ...
          100*mean(corloc.corloc_1), 100*std(corloc.corloc_1));
  fprintf('Average: detRate_nu: %.2f / %.2f ; detRate_5: %.2f / %.2f ; detRate_1: %.2f / %.2f\n', ...
          100*mean(detRate.detRate_nu), 100* std(detRate.detRate_nu), ...
          100*mean(detRate.detRate_5), 100*std(detRate.detRate_5), ...
          100*mean(detRate.detRate_1), 100*std(detRate.detRate_1));
  corloc_nu = [corloc_nu; corloc.corloc_nu];
  corloc_5 = [corloc_5; corloc.corloc_5];
  corloc_1 = [corloc_1; corloc.corloc_1];
  detRate_nu = [detRate_nu; detRate.detRate_nu];
  detRate_5 = [detRate_5; detRate.detRate_5];
  detRate_1 = [detRate_1; detRate.detRate_1];
  % save results
  if opts.save_result
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
                'num_iters', ...
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

function [proposals, bboxes, imdb_path] = load_imdb(opts)
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
end

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

function [sol_buffer] = get_solutions(opts, sol_buffer, l, r)
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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x_opt_nu, x_opt_5, x_opt_1, e_opt_1] = process_solutions(opts, iter)
  n = opts.n; nu = opts.nu; tau = opts.tau; nms_IoU = opts.nms_IoU;
  x_opt_nu = opts.sol_buffer.x_s{iter};
  x_opt_5 = post_proc_nms(opts.DL, opts.sol_buffer.x_s{iter}, opts.sol_buffer.e_s{iter}, ...
                          5, opts.proposals, tau, 1, nms_IoU);
  [x_opt_1, e_opt_1] = post_proc_nms(opts.DL, opts.sol_buffer.x_s{iter}, opts.sol_buffer.e_s{iter}, ...
                          1, opts.proposals, tau, 1, nms_IoU);
end 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = save_result(opts, corloc, detRate, sol)
  fprintf('Saving result ... ');
  mkdir(opts.save_path);
  save(fullfile(opts.save_path, sprintf('x_opt_raw_IoU_%.2f_%d_solutions.mat', opts.nms_IoU, opts.num_iters)), ...
       '-struct', 'sol');
  save(fullfile(opts.save_path, sprintf('corloc_raw_IoU_%.2f_%d_solutions.mat', opts.nms_IoU, opts.num_iters)), ...
       '-struct', 'corloc');
  save(fullfile(opts.save_path, sprintf('detection_rate_raw_IoU_%.2f_%d_solutions.mat', opts.nms_IoU, opts.num_iters)), ...
       '-struct', 'detRate');
  fprintf('DONE!\n');
end
