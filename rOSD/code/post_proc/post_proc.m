function [x_opt, e_opt] = post_proc(data_loader, x, e, k, ...
                                    num_nb, num_nb_regions, ...
                                    group_ids, x_weight)
% POST_PROC
% [x_opt, e_opt] = post_proc(data_loader, x, e, k, ...
%                            num_nb, num_nb_regions, ...
%                            group_ids, x_weight)
%
% Perform post processing ensemble.
%
% Parameters:
%
%   data_loader: an instance of DataLoader class.
%
%   x: (n x 1) cell, x{i} contains indices of candidate regions
%      in image i.
%
%   e: (n x 1) cell, e{i} contains indices of candidate neighbors
%      of image i.
%
%   k: int, maximum number of region retained in each image.
%
%   num_nb: int, maximum number of neighbors of each image.
%
%   num_nb_regions: int, maximum number of regions in neighboring 
%                   images used in computing the score of each region.
%
%   group_ids: (n x 1) cell, group_ids{i} contains the group id of 
%              proposals in image i.
%
%   x_weight: (n x 1) cell, region_weights{i} contains weights 
%             for regions in image i.
%
% Returns
%
%   x_opt: (n x 1) cell, x_opt{i} contains the indices of retained 
%           regions in image i after the processing.
%
%   e_opt: (n x 1) cell, e_opt{i} contains the indices of neighbors
%          of image i after the processing.
% 

n = size(x, 1);
if ~exist('num_nb', 'var')
  num_nb = Inf;
end
if ~exist('num_nb_regions', 'var')
  num_nb_regions = Inf;
end
if ~exist('x_weight', 'var')
  x_weight = cell(n, 1);
  for i = 1:n
    x_weight{i} = ones(1, numel(x{i}));
  end
end

x_opt = cell(n, 1);
if k == 1
  e_opt = cell(n,1);
else 
  e_opt = [];
end


for i = 1:n
  sim_to_neighbors = zeros(numel(x{i}), numel(e{i}));
  for j = 1:numel(e{i})
    S = get_S(data_loader, i, e{i}(j));
    x_pair_weight = transpose(x_weight{i}) * x_weight{e{i}(j)};
    sim_to_neighbors(:,j) = sum_row_k(x_pair_weight .* S(x{i}, x{e{i}(j)}), ...
                                      num_nb_regions);
  end
  region_score = zeros(numel(x{i}), 1);
  for j = 1:numel(x{i})
    [~,ids] = sort(sim_to_neighbors(j,:), 'descend');
    chosen_ids = ids(1:min(num_nb, numel(e{i})));
    region_score(j) = sum(sim_to_neighbors(j,chosen_ids));
  end
  [~, region_ids] = sort(region_score, 'descend');
  x_opt{i} = nms_x(x{i}(region_ids), group_ids{i}, k);
  if k == 1
    [~, neighbor_ids] = sort(sim_to_neighbors(region_ids(1),:), 'descend');
    e_opt{i} = e{i}(neighbor_ids(1:min(num_nb, numel(e{i}))));
  end
end


end

%-------------------------------------------
function [x_s] = nms_x(top_ids, group_ids_list, nu)
  % efficient if nu is small (which is usually the case)
  x_s = [];
  for iter = 1:numel(top_ids) 
    if ismember(group_ids_list(top_ids(iter)), group_ids_list(x_s))
      continue;
    end
    x_s = [x_s, top_ids(iter)];
    if numel(x_s) == nu 
      break;
    end
  end
end
