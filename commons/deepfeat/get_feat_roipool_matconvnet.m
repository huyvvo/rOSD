root = fullfile(DATA_ROOT, imgset);
images_root = fullfile(DATA_ROOT, images_imgset);
classes = get_classes(imgset);

% Run matconvnet setup, please change the location as your own
run(fullfile(MATCONVNET_PATH, 'matlab/vl_setupnn.m'));

vgg_path = fullfile(VGG_DIR, 'imagenet-vgg-verydeep-19.mat');

% Run matconvnet setup, please change the location as your own
run('~/code/matconvnet-1.0-beta25/matlab/vl_setupnn.m');

% Specify which layer to use
% "37"/"28"/"19" for the layer relu5_4/relu4_4/relu3_4 of vgg19, 
% note that the index of layer relu5_4 in vgg19 is 36 but res(i+1).x is the output of layer i. 
opt = struct('layer', 37, 'model', 'imagenet-vgg-verydeep-19.mat', ...
             'wsize', 7, 'feat_type', 'vgg19_relu54_matconvnet_77_roi_pooling_noresize', ...
             'bboxes_transform', 1/16); 


net = load(vgg_path);
% Remove FC layers and transfer to GPU
net.layers(opt.layer+1:end)=[];
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu');
  
cl = numel(classes);
clname = classes{cl};
save_path = fullfile(root, clname, 'features/proposals', opt.feat_type);
mkdir(save_path);
imdb = load(fullfile(root, clname, [clname, '_lite.mat']));
imdb.images = getfield(load(fullfile(images_root, clname, [clname, '.mat'])), 'images');
averageImage = net.meta.normalization.averageImage;
n = size(imdb.images, 1);
begin_time = tic;
for i = 1:n
  if mod(i,100) == 1
    fprintf('Computing %d, %.2f secs elapsed\n', i, toc(begin_time));
  end
  im = imdb.images{i};
  im_ = single(im);
  [h,w,c] = size(im_);
  if  c > 2
      im_ = im_ - imresize(averageImage,[h,w]) ;
  else    
      im_ = bsxfun(@minus,im_,imresize(averageImage,[h,w])) ;
  end
  res = vl_simplenn(net, gpuArray(im_));
  feat = res(opt.layer).x;
  % get ROI features
  boxes = transpose(imdb.proposals{i});
  rois = [ones(1,size(boxes,2)) ; boxes];
  roi_feat = gather(vl_nnroipool(feat, gpuArray(single(rois)), 'Subdivisions', [opt.wsize opt.wsize], ...
                                 'Transform', opt.bboxes_transform));
  data = permute(roi_feat, [4,3,1,2]);
  save(fullfile(save_path, sprintf('%d.mat', i)), 'data');
end
fprintf('\n');