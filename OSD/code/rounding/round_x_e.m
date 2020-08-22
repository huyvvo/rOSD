function [ x_s, e_s ] = round_x_e(data_loader, x_run, e_run, e_candidate, ...
                                  nu, tau, row_order)
% ROUND_X_E
% [ x_s, e_s ] = round_x_e(data_loader, x_run, e_run, e_candidate, ...
%                          nu, tau, row_order)
% 
% Rounding the running average (x,e) to an admissible integer solution.
% x is rounded before e.
%
% Parameters:
%
%   data_loader: an instance of DataLoader class.
%
%   x_run: (n x 1) cell, x{i} is a (1 x num_regions(i)) matrix containing
%      real scores of proposals in image i.
%
%   e_run: (n x n) matrix, e(i,j) is the score of the link from image i 
%      to image j.
%
%   e_candidate: (n x 1) cell, e_candidate{i} contains indices of 
%                possible neighbors of image i.
%
%   nu: int, maximum number of proposals retained in each image.
%
%		tau: int, maximum number of neighbors of each image.
%
%		row_order: (1 x n) array, order in which rows of x are processed.
%
% Returns:
%
%   x_s: (n x 1) cell, x after rounded, x{i} contains indices of
%        retained proposals in image i.
%   
%   e_s: (n x 1) cell, e after rounded, e{i} contains indices of 
%        neighbors of image i.
%

if ~exist('row_order', 'var')
	row_order = 1:size(x_run, 1);
end

x_s = round_x(data_loader, x_run, e_run, e_candidate, nu, row_order);
e_s = ascent_e(data_loader, x_s, e_candidate, tau);

end
