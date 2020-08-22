function [] = visualize_cnn_features(feat, save_path)
  if numel(size(feat)) == 3
    heatmap = squeeze(sum(feat));
  elseif numel(size(feat)) == 2 && sum(size(feat)>1) == 2
    heatmap = feat;
  else
    error(sprintf('Feat must have dim=2 or dim=3.', numel(size(feat)))); 
  end
  clf; colormap('summer');
  imagesc(heatmap); colorbar;
  axis image
  saveas(gcf, save_path);
end