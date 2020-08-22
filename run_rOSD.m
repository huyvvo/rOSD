addpath(genpath(fullfile(ROOT, 'rOSD')));
addpath(genpath(fullfile(ROOT, 'commons')));

imgset = 'vocx_cnn';
classes = get_classes(imgset);
all_classes = 1:numel(classes)-1;
big_classes = [numel(classes)];

% compute solutions
cd(fullfile(ROOT, 'rOSD/scripts/solutions'));
class_indices = all_classes;
solution_indices = 1:10;
script;

% evaluate solutions
fprintf('Mean single-object colocalization performance on %s: %.1f +- %.1f\n', ...
        imgset, ...
        100*mean(mean(cl1(1:numel(classes)-1,:))), ...
        100*std(mean(cl1(1:numel(classes)-1,:))));

fprintf('Mean single-object discovery performance on %s: %.1f +- %.1f\n', ...
        imgset, ...
        100*mean(cl1(end,:)), ...
        100*std(cl1(end,:)));
