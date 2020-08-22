function [dr, dc] = detection_rate(proposals, bboxes, x, threshold)
% DETECTION_RATE
%
% [ dr ] = detection_rate(proposals, bboxes, threshold)
%
% Compute detection rate.
%
% Parameters:
%
%   proposals: (n x 1) cell, proposals{i} contains the proposals 
%              of image i. Proposals are in the format 
%              [xmin, ymin, xmax, ymax] where (xmin, ymin) is the 
%              coordinate of the top left corner, (xmax, ymax) is
%              the coordinate of the bottom right corner.
%
%   bboxes: (n x 1) cell, bboxes{i} contains ground truth bboxes
%           of image i. Bboxes are in the same format as proposals.
%
%   x:  (n x 1) cell, x{i} contains indices of regions retained in
%       in image i.
%
%   threshold: double, threshold for IOU score.
%
%
% Returns:
%
%   dr: double, detection rate.
%   
%   dc: array, detection count.

[~, iou] = CorLoc(proposals, bboxes, x, threshold);
dc = arrayfun(@(i) sum(max(transpose(iou{i})) >= threshold), 1:numel(proposals));
dr = sum(dc) / sum(cellfun(@numel, bboxes)) * 4;

end

