function [ x ] = ascent_x(data_loader, x, e, e_candidate, nu, ...
                          num_regions, group_ids, row_order)
% ASCENT_X
% [ x ] = ascent_x(data_loader, x, e, e_candidate, nu, ...
%                  num_regions, group_ids, row_order)
% 
% Perform coordinate ascent on x.
%
% Parameters:
%
%   data_loader: an instance of DataLoader class.
%
%   x: (n x 1) cell, x{i} represents current retained proposals
%      in image i.
%
%   e: (n x 1) cell, e{i} represents current neighbors of image i.
%
%   e_candidate: (n x 1) cell, e_candidate{i} contains indices
%                of possible neighbors of image i. 
%
%   nu: int, maximum number of proposals retained in each image.
%
%   num_regions: (n x 1) array, number of proposals in each image.
%
%   group_ids: (n x 1) cell, group_ids{i} contains the group id of 
%              proposals in image i.
%
%   row_order: (1 x K) array, indicating which rows of x will be updated
%              and in which order they are updated.  
%
% Returns:
%
%   x: (n x 1) cell, x after performing coordinate ascent.
%

n = size(x,1);
if ~exist('row_order', 'var')
    row_order = 1:n;
end

assert(size(row_order, 2) == n & size(row_order, 1) == 1);
for i = row_order
  Sx_sum = zeros(num_regions(i),1);
  for j = e_candidate{i}
    linked_level = sum(ismember(i, e{j})) + sum(ismember(j, e{i}));
    if i ~= j & linked_level > 0
      current_S = get_S(data_loader, i, j);
      Sx_sum = Sx_sum + linked_level*sum(current_S(:, x{j}), 2); 
    end    
  end
  x{i} = nms_x(Sx_sum', group_ids{i}, nu);
end

end 

%-------------------------------------------
function [x_s] = nms_x(scores, group_ids, nu)
  A = [scores', group_ids'];
  [A, ids] = sortrows(A,'descend');
  [~,ia] = unique(A(:,2), 'stable');
  x_s = ids(ia(1:min(nu,end)))';
end

