function [] = vis_localization(dataset_name, save_path, num_images)
% [] = vis_localization(image_path, dataset_name, save_path)

fout = fopen(save_path, 'w');
fprintf(fout, ...
    ['<html><head>', ...
     sprintf('<title>Localization results for %s </title>', dataset_name), ...
     '</head>\n']);

for i = 1:num_images
    fprintf(fout, ['<table border="0">\n']);
    fprintf(fout, ['\t<tr>\n']);
    fprintf(fout, ...
     sprintf('\t\t<font size=5> Image %d </font>\n', i));
    fprintf(fout, ['\t</tr>\n']);
    fprintf(fout, ['\t<tr>\n']);
    fprintf(fout, ...
    [sprintf('\t\t<td valign=top>\n'), ...
     sprintf('\t\t\t<font size=4> Original image </font>\n'), ...
     sprintf('\t\t\t<br>\n'), ...
     sprintf('\t\t\t<img src="%s" width="280" border="1">\n', ...
        fullfile('.', sprintf('%s_ori_%04d.jpg', dataset_name, i))), ...
     sprintf('\t\t</td>\n')] ...
     );

    fprintf(fout, ...
    [sprintf('\t\t<td valign=top>\n'), ...
     sprintf('\t\t\t<font size=4> Candidate localizations </font>\n'), ...
     sprintf('\t\t\t<br>\n'), ...
     sprintf('\t\t\t<img src="%s" width="280" border="1">\n', ...
        fullfile('.', sprintf('%s_candidate_%04d.jpg', dataset_name, i))), ...
     sprintf('\t\t</td>\n')] ...
    );

    fprintf(fout, ...
    [sprintf('\t\t<td valign=top>\n'), ...
     sprintf('\t\t\t<font size=4> Final localizations </font>\n'), ...
     sprintf('\t\t\t<br>\n'), ...
     sprintf('\t\t\t<img src="%s" width="280" border="1">\n', ...
        fullfile('.', sprintf('%s_final_%04d.jpg', dataset_name, i))), ...
     sprintf('\t\t</td>\n')] ...
    );

    fprintf(fout, [sprintf('\t</tr>\n'), '</table>\n']);
end


fprintf(fout, ['</html>\n'])
fclose(fout);


end