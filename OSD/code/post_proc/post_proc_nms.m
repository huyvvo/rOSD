function [x_opt, e_opt] = post_proc_nms(data_loader, x, e, k, ...
                                        proposals, num_nb, num_nb_regions, ...
                                        nms_IoU, x_weight)
% POST_PROC_NMS
% [x_opt, e_opt] = post_proc_nms(data_loader, x, e, k, ...
%                                proposals, num_nb, num_nb_regions, ...
%                                nms_IoU, x_weight)
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
%   proposals: (n x 1) cell, proposals{i} contains the coordinates
%              of proposals in image i.
%
%   num_nb: int, maximum number of neighbors of each image.
%
%   num_nb_regions: int, maximum number of regions in neighboring 
%                   images used in computing the score of each region.
%
%   nms_IoU: double, IoU threshold used in non-maximum suppression.
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
  x_opt{i} = x{i}(region_ids);  
  if k == 1
    [~, neighbor_ids] = sort(sim_to_neighbors(region_ids(1),:), 'descend');
    e_opt{i} = e{i}(neighbor_ids(1:min(num_nb, numel(e{i}))));
  end
end

x_opt = nms(x_opt, proposals, nms_IoU, k);

end