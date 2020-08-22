% imgset = 'vocx_vgg19_matconvnet_cnn_03_20_1_05';
imgset = 'imagenet6_wei_resizemax500_ssfast';
% imgset = 'voc07_trainval_full_vgg16_cnn_03_20_1_05_with_roots';
% imgset = 'voc07_test_full_vgg16_cnn_03_50_1_05_with_roots'
% imgset = 'imagenet6_200_vgg16_cnn_03_20_1_05';

root = ['~/', imgset];
classes = get_classes(imgset);
fprintf('ROOT: %s\n', root);

suffix = '_lite';
verbose = false;

num_props = [];
corloc05 = [];
corloc07 = [];
corloc09 = [];
numpos05 = [];
numpos07 = [];
numpos09 = [];
pos_box_percentage = [];
mean_num_box_retrieved = [];
sum_num_box_retrieved = [];
percentage_box_retrieved = [];
num_boxes = [];
for cl = 7
  clname = classes{cl};
  imdb = load(fullfile(root, clname, [clname, suffix, '.mat']));
  
  proposals = imdb.proposals;
  bboxes = imdb.bboxes;

  num_pos_boxes = count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.5)';
  num_proposals = cellfun(@numel, proposals)/4;

  num_objects = cellfun(@(x) size(x,1), bboxes);
  [~, iou] = CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.5);
  num_objects_retrieved = cellfun(@(x) sum(max(x') >= 0.5), iou);

  num_props = [num_props mean(num_proposals)];
  corloc05 = [corloc05 CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.5)];
  corloc07 = [corloc07 CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.7)];
  corloc09 = [corloc09 CorLoc(proposals, bboxes, create_x_all_regions(proposals), 0.9)];
  numpos05 = [numpos05 mean(count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.5))];
  numpos07 = [numpos07 mean(count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.7))];
  numpos09 = [numpos09 mean(count_valid_boxes(proposals, bboxes, create_x_all_regions(proposals), 0.9))];
  pos_box_percentage = [pos_box_percentage mean(num_pos_boxes./num_proposals')*100];
  mean_num_box_retrieved = [mean_num_box_retrieved mean(num_objects_retrieved)];
  sum_num_box_retrieved = [sum_num_box_retrieved sum(num_objects_retrieved)];
  percentage_box_retrieved = [percentage_box_retrieved mean(num_objects_retrieved./num_objects)];
  num_boxes = [num_boxes sum(num_objects)];
  
  
  if verbose
    fprintf('Processing for class %s\n', clname);
    fprintf('Average number of proposals: %.2f\n', num_props(end));
    fprintf('CorLoc 05/07/09: %.4f/%.4f/%.4f\n', corloc05(end), corloc07(end), corloc09(end));
    fprintf('Number of positive boxes 05/07/09: %.2f/%.2f/%.2f\n', numpos05(end), numpos07(end), numpos09(end));
    fprintf('Percentage of positive boxes: %.2f%%\n', pos_box_percentage(end));
    fprintf('Average number of objects retrieved: %.2f\n', mean_num_box_retrieved(end));
    fprintf('Number of objects retrieved: %d\n', sum_num_box_retrieved(end));
    fprintf('Average percentage of objects retrieved: %.2f\n', percentage_box_retrieved(end));
    fprintf('Total number of objects: %d\n', num_boxes(end));
  end
end
fprintf('Average num_props: %.4f\n', mean(num_props));
fprintf('Average corloc05/07/09: %.4f/%.4f/%.4f\n', mean(corloc05), mean(corloc07), mean(corloc09));
fprintf('Average numpos05/07/09: %.4f/%.4f/%.4f\n', mean(numpos05), mean(numpos07), mean(numpos09));
fprintf('Average percentage pos boxes: %.2f%%\n', mean(pos_box_percentage));
fprintf('Total number of objects retrieved: %d\n', sum(sum_num_box_retrieved));
fprintf('Average percentage of objects retrieved: %.2f\n', mean(percentage_box_retrieved));
fprintf('Total number of objects: %d\n', sum(num_boxes));
