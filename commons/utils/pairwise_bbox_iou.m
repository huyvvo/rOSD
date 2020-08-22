function [IoU] = pairwise_bbox_iou(bboxA, bboxB)
  IoU = {};
  for i = 1:size(bboxA,1)
    IoU{end+1,1} = bbox_iou(bboxA(i,:), bboxB)';
  end
  IoU = cell2mat(IoU);
end