function [] = visualize_result(imdb, save_path, x_combined, ...
                               x_opt_nu, x_opt_1)
% VISUALIZE_RESULT
%
% visualize_result(imdb, save_path, x_combined, ...
%                  x_opt_nu, x_opt_1)
%
% Parameters:
%
%   imdb: string or struct, path to the imdb or the imdb itself,
%         containing fields 'images', 'proposals', 'bboxes'
%
%   save_path: string, folder to save visualization
%
%   fn_pattern: string, filename pattern for images
%
%   x_combined: (n x 1) cell representing retained proposals 
%               before EM
%
%   x_opt_nu: (n x 1) cell representing top \nu proposals by EM
%
%   x_opt_1: (n x 1) cell representing top 1 proposal by EM
%

%-----------------------------------------------------------------
% CREATE IMAGES

if strcmp(class(imdb), 'char') || strcmp(class(imdb), 'string')
  imdb = load(imdb);
end
n = size(imdb.images, 1);
mkdir(save_path);
for i = 1:n
  if numel(x_opt_nu) < i | isempty(x_opt_nu{i})
    continue;
  end
  img_box = imdb.images{i};
  imwrite(img_box, fullfile(save_path, ...
                    sprintf('%d_ori.jpg', i)), 'jpg');

  bboxes = imdb.bboxes{i};
  for idx = 1:size(bboxes, 1)
    bbox = bboxes(idx, :);
    img_box = drawBox(img_box, bbox, 4, [0, 0, 0]);
    img_box = drawBox(img_box, bbox, 2, [255, 255, 255]);
  end
  imwrite(img_box, fullfile(save_path, ...
                    sprintf('%d_ori_gtboxes.jpg', i)), 'jpg');

  combined_bboxes = imdb.proposals{i}(x_combined{i}, :);
  combined_img_box = img_box;
  for idx = 1:size(combined_bboxes, 1)
    bbox = combined_bboxes(idx, :);
    combined_img_box = drawBox(combined_img_box, bbox, 6, [100, 0, 0]);
    combined_img_box = drawBox(combined_img_box, bbox, 3, [255, 0, 0]);
  end
  imwrite(combined_img_box, fullfile(save_path, ...
                    sprintf('%d_combined.jpg', i)), 'jpg');

  
  topnu_img_box = img_box;
  topnu_bboxes = imdb.proposals{i}(x_opt_nu{i}, :);
  for idx = 1:size(topnu_bboxes, 1)
    bbox = topnu_bboxes(idx, :);
    topnu_img_box = drawBox(topnu_img_box, bbox, 6, [100, 0, 0]);
    topnu_img_box = drawBox(topnu_img_box, bbox, 3, [255, 0, 0]);
  end
  imwrite(topnu_img_box, fullfile(save_path, ...
                          sprintf('%d_topnu.jpg', i)), 'jpg');

  top1_img_box = img_box;
  top1_bboxes = imdb.proposals{i}(x_opt_1{i}, :);
  for idx = 1:size(top1_bboxes, 1)
    bbox = top1_bboxes(idx, :);
    top1_img_box = drawBox(top1_img_box, bbox, 6, [100, 0, 0]);
    top1_img_box = drawBox(top1_img_box, bbox, 3, [255, 0, 0]);
  end
  imwrite(top1_img_box, fullfile(save_path, ...
                          sprintf('%d_top1.jpg', i)), 'jpg');


end

%----------------------------------------------------------------
% CREATE WEBPAGE

iwidth_img  = 280;
iheight_img = 200;

fout = fopen(fullfile(save_path, 'index.html'), 'w');
fprintf(fout, '<html><head><title>Visualization</title></head>\n');
fprintf(fout, '<br><br><br>\n');

for i = 1:n
  if numel(x_opt_nu) < i | isempty(x_opt_nu{i})
    continue;
  end
  % start table
  fprintf(fout, '<table border="0">\n');
  fprintf(fout, '<tr>\n');

  % original image
  img_name = fullfile(sprintf('%d_ori.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, sprintf('<font size=5>Original image %d</font>', i));
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % original image with ground truth boxes
  img_name = fullfile(sprintf('%d_ori_gtboxes.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, '<font size=5>Original image with true boxes</font>');
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % boxes retained before the ensemble method
  img_name = fullfile(sprintf('%d_combined.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, '<font size=5>Boxes retained before EM</font>');
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % top nu boxes
  img_name = fullfile(sprintf('%d_topnu.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, '<font size=5>Top nu boxes with EM</font>');
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % top1 box
  img_name = fullfile(sprintf('%d_top1.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, '<font size=5>Top 1 box with EM</font>');
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>');

  % end table
  fprintf(fout, '</tr>\n');
  fprintf(fout, '</table>\n');
  fprintf(fout, '<br><br><br>\n');
end
fprintf(fout, '</html>\n');
fclose(fout);
