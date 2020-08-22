function [] = visualize_neighbors(save_path, images, neighbors)
% VISUALIZE_NEIGHBORS
%
% visualize_neighbors(save_path, images, neighbors)
%
% A function for visualizing images and their neighbors.
%
% Parameters:
%
%   save_path: string, folder to save visualization.
%
%   images: (n x 1) cell, each cell contains an image.
%
%   neighbors: (n x 1) cell, neighbors{i} is an array 
%              containing neighbors of image i.
%

%-----------------------------------------------------------------
% SAVE IMAGES

mkdir(save_path);
for i = 1:size(images, 1)
  imwrite(images{i}, fullfile(save_path, sprintf('%d.jpg', i)), 'jpg');
end

%----------------------------------------------------------------
% CREATE WEBPAGE

iwidth_img  = 280;
iheight_img = 200;

fout = fopen(fullfile(save_path, 'index.html'), 'w');
fprintf(fout, '<html><head><title>Visualization</title></head>\n');
fprintf(fout, '<br><br><br>\n');

for i = 1:size(images, 1)
  if isempty(neighbors{i})
    continue;
  end
  % start table
  fprintf(fout, '<table border="0">\n');
  fprintf(fout, '<tr>\n');

  % First image
  img_name = fullfile(sprintf('%d.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, sprintf('<font size=5>Row %d, image %d</font>', i, 1));
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>\n');

  for j = neighbors{i}
    img_name = fullfile(sprintf('%d.jpg', j));
    fprintf(fout, '<td valign=top>');
    fprintf(fout, sprintf('<font size=5>image %d</font>', j)); 
    fprintf(fout, '<br>');
    fprintf(fout, ['<img src="', img_name, '" width="', ...
                   num2str(iwidth_img), '" border="1"></a>']);  
    fprintf(fout, '</td>\n');
  end

  % end table
  fprintf(fout, '</tr>\n');
  fprintf(fout, '</table>\n');
  fprintf(fout, '<br><br><br>\n');
end
fprintf(fout, '</html>\n');
fclose(fout);
