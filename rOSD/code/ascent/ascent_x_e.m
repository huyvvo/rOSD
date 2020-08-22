function [x,e] = ascent_x_e(data_loader, x, e, e_candidate, nu, tau, ...
                            num_regions, group_ids, num_iter, xrows)
% ASCENT_X_E
% [x,e] = ascent_x_e(data_loader, x, e, e_candidate, nu, tau, ...
%                    num_regions, group_ids, num_iter, xrows)
%
% Block coordinate ascent: Iteratively update (x,e) to maximize 
% objective function.
%
% Parameters:
%
%   data_loader: instance of class DataLoader.
%
%   x: (n x 1) cell, elements in x{i} are indices of proposals
%      in image i.
%
%   e: (n x 1) cell, e{i} represents current neighbors of image i.
%
%   e_candidate: (n x 1) cell, e_candidate{i} contains indices
%                of possible neighbors of image i. 
%
%   nu: int, maximum number of proposals retained in each image.
%
%   tau: int, maximum number of neighbors of each image.
%
%   num_regions: (n x 1) array, number of proposals in each image.
%
%   group_ids: (n x 1) cell, group_ids{i} contains the group id of 
%              proposals in image i.
%
%   num_iter: int, number of ascent iterations.
%
%   xrows: (1 x K) array, indicating which rows of x will be updated
%          and in which order they are updated.  
%
% Returns:
%
%   values of x and e after the block coordinate ascent.
%

n = size(x, 1);
if ~exist('xrows', 'var')
  xrows = 1:n;
end
for i = 1:num_iter
  x = ascent_x(data_loader, x, e, e_candidate, nu, num_regions, group_ids, xrows);
  e = ascent_e(data_loader, x, e_candidate, tau);
end

end