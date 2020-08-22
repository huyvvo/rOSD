function [img_box] = visualize_proposals(img_box, ...
                                  bboxes, proposals, save_path, ...
                                  draw_boxes, draw_props, ...
                                  width_boxes, width_props, ...
                                  color_boxes, color_props)
% VISUALIZE_PROPOSALS
%
% visualize_proposals(image, bboxes, proposals, ...
%                     save_path, draw_boxes, draw_props, ...
%                     width_boxes, width_props)
%
% Parameters:
%
%   img_box: an image object
%
%   bboxes:
%
%   proposals:
%
%   save_path:
%
%   draw_boxes: bool, default is true
%
%   draw_props: bool, default is true
%
%   width_boxes: int, default is 2
%
%   width_props: int, default is 3
%

if exist('draw_boxes', 'var') == 0
  draw_boxes = true;
end 
if exist('draw_props', 'var') == 0
  draw_props = true;
end
if exist('width_boxes', 'var') == 0
  width_boxes = 2;
end
if exist('width_props', 'var') == 0
  width_props = 3;
end
if exist('color_boxes', 'var') == 0
  color_boxes = [255,255,255];
end
if exist('color_props', 'var') == 0
  color_props = [255,0,0];
end

if draw_boxes
  for idx = 1:size(bboxes, 1)
    bbox = bboxes(idx, :);
    img_box = drawBox(img_box, bbox, 2*width_boxes, [0, 0, 0]);
    img_box = drawBox(img_box, bbox, width_boxes, color_boxes);
  end
end

if draw_props
  for idx = 1:size(proposals, 1)
    bbox = proposals(idx, :);
    img_box = drawBox(img_box, bbox, 2*width_props, [0, 0, 0]);
    img_box = drawBox(img_box, bbox, width_props, color_props);
  end
end

imwrite(img_box, save_path, 'jpg');

