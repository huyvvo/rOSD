function [class_indices] = get_class_indices(root, imgset)
% GET_CLASS_INDICES
%
% [class_indices] = get_class_indices(imgset)

if numel(imgset) >= 2 && strcmp(imgset(1:2), 'od')
  class_indices = getfield(load(fullfile(root, 'commons/class_indices/od.mat')), 'class_indices');
elseif length(imgset) >= 4 & strcmp(imgset(1:4), 'vocx')
  class_indices = getfield(load(fullfile(root, 'commons/class_indices/vocx.mat')), 'class_indices');
elseif numel(imgset) >= 3 && strcmp(imgset(1:3), 'voc')
  class_indices = getfield(load(fullfile(root, 'commons/class_indices/voc.mat')), 'class_indices');
elseif numel(imgset) >= 4 && strcmp(imgset(1:4), 'coco')
  class_indices = getfield(load(fullfile(root, 'commons/class_indices/voc12.mat')), 'class_indices');
else
  error('Invalid value of "imgset"');
end

end
