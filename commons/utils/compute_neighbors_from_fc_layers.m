root = fullfile(DATA_ROOT, imgset);
feat_name = 'vgg19_fc6_matconvnet_resize';
similarity_measure = 'cos';
num_neighbors = 50;

clname = 'mixed';
imdb = load(fullfile(root, clname, [clname, '_lite.mat']));
n = numel(imdb.bboxes);

feats = cell(n,1);
for i = 1:n 
  feat = getfield(load(fullfile(root, clname, 'features/image', feat_name, sprintf('%d.mat', i))), 'data');
  feats{i} = mean(reshape(feat, size(feat,1), []), 2)';
end
feats = cell2mat(feats);

if strcmp(similarity_measure, 'cos')
  norm_feats = feats ./ sqrt(sum(feats.*feats, 2));
  similarity = norm_feats * norm_feats';
  similarity(sub2ind([n,n], 1:n, 1:n)) = -1;
elseif strcmp(similarity_measure, 'l2')
  norm_2 = sum(feats .* feats, 2);
  DIST = repmat(norm_2, 1, n) + repmat(norm_2', n, 1) - 2*feats*feats';
  similarity = (1 + max(DIST(:))) - DIST;
  similarity(sub2ind([n,n], 1:n, 1:n)) = -1;
end

e = cell(n,1);
for i = 1:n 
  [~, max_idx] = sort(similarity(i,:), 'descend');
  e{i} = max_idx(1:min(end-1, num_neighbors));
end

save_path = fullfile(root, sprintf('neighbor_%s_%s', similarity_measure, feat_name), clname);
mkdir(save_path);
save(fullfile(save_path, sprintf('%d.mat', num_neighbors)), 'e');
