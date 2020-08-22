function [ e ] = ascent_e( data_loader, x, e_candidate, tau)
% ASCENT_E
% [ e ] = ascent_e(data_loader, x, e_candidate, tau)
% 
% Perform coordinate ascent on e while fixing x.
%
% Parameters:
%
%   data_loader: an instance of DataLoader class.
%
%   x: (n x 1) cell, elements in x{i} are indices of proposals
%      in image i.
%
%   e_candidate: (n x 1) cell, e_candidate{i} contains indices
%                of possible neighbors of image i. 
%
%   tau: int, the maximum number of neighbors of each image.
%
% Returns:
%
%   e: (n x 1) cell, e after performing coordinate ascent.
%


n = size(x,1);
e = cell(n,1);
% compute similarity between pairs of images
A = sparse(n,n);
for i = 1:n
  for j = e_candidate{i}
    if A(j,i) ~= 0
      A(i,j) = A(j,i);
    else
      current_S = get_S(data_loader, i, j);
      try
        A(i,j) = sum(sum(current_S(x{i}, x{j})));
      catch exception
        A(i,j) = x{i}*current_S*transpose(x{j});
      end
    end
  end
end

for i = 1:n
  line_weight = A(i,:);
  line_weight(e_candidate{i}) = line_weight(e_candidate{i}) + 1;
  [ ~, idx_top ] = sort(line_weight, 'descend');
  e{i} = idx_top(1:min(tau, numel(e_candidate{i})));
end
