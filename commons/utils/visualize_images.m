function [] = visualize_images(images, save_path)
% VISUALIZE_IMAGES
%
% visualize_images(images, save_path)
%
% A general function for visualizing images in a web page.
%
% Parameters:
%
%   images: (n x 1) cell, each cell contains some images. Images 
%           in a same row in the cell are to be visualized in a same 
%           row in the webpage.
%
%   save_path: string, folder to save visualization
%

%-----------------------------------------------------------------
% SAVE IMAGES

mkdir(save_path);
for i = 1:size(images, 1)
  for j = 1:numel(images{i})
    imwrite(images{i}{j}, fullfile(save_path, ...
                      sprintf('%d_%d.jpg', i, j)), 'jpg');
  end
end

%----------------------------------------------------------------
% CREATE WEBPAGE

iwidth_img  = 280;
iheight_img = 200;

fout = fopen(fullfile(save_path, 'index.html'), 'w');
fprintf(fout, '<html><head><title>Visualization</title></head>\n');
fprintf(fout, '<br><br><br>\n');

for i = 1:size(images, 1)
  % start table
  fprintf(fout, '<table border="0">\n');
  fprintf(fout, '<tr>\n');

  % First image
  img_name = fullfile(sprintf('%d_1.jpg', i));
  fprintf(fout, '<td valign=top>');
  fprintf(fout, sprintf('<font size=5>Row %d, image %d</font>', i, 1));
  fprintf(fout, '<br>');
  fprintf(fout, ['<img src="', img_name, '" width="', ...
                 num2str(iwidth_img), '" border="1"></a>']);  
  fprintf(fout, '</td>\n');

  for j = 2:numel(images{i})
    img_name = fullfile(sprintf('%d_%d.jpg', i, j));
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
