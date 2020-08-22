function [ x_s ] = round_x(data_loader, x, e, e_candidate, nu, row_order)
% ROUND_X
% [ x_s ] = round_x(data_loader, x, e, e_candidate, nu, row_order)
% 
% Rounding real-valued x to get integer-valued x_s.
%
% Parameters:
%
%   data_loader: an instance of DataLoader class.
%
%   x: (n x 1) cell, x{i} is a (1 x num_regions(i)) matrix containing
%      real scores of proposals in image i.
%
%   e: (n x n) matrix, e(i,j) is the score of the link from image i 
%      to image j.
%
%   e_candidate: (n x 1) cell, e_candidate{i} contains indices of 
%                possible neighbors of image i.
%
%   nu: int, the maximum number of proposals retained in each image.
%
%   row_order: (1 x n) matrix, the order in which rows of x
%              are processed.
%
% Returns:
%
%   x_s: (n x 1) cell, x after performing coordinate ascent
%

n = size(x,1);
if ~exist('row_order', 'var')
    row_order = 1:n;
end
assert(size(row_order, 2) == n & size(row_order, 1) == 1);

num_regions = cellfun(@numel, x);
x_s = cell(n,1);
processed = zeros(1, n); % mark rows of x that have been rounded
for i = row_order
  Sx_sum = zeros(num_regions(i),1);
  for j = e_candidate{i}
    current_S = get_S(data_loader, i, j);
    if processed(j) == 0
      Sx_sum = Sx_sum + (e(i,j)+e(j,i))*current_S*transpose(x{j});
    else
      Sx_sum = Sx_sum + (e(i,j)+e(j,i))*sum(current_S(:, x_s{j}), 2);
    end      
  end
  [~, idx_top] = sort(Sx_sum, 'descend');
  x_s{i} = transpose(idx_top(1:min(nu, num_regions(i))));
  processed(i) = 1;
end
