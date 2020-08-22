root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

% Run matconvnet setup, please change the location as your own
run(fullfile(MATCONVNET_PATH, 'matlab/vl_setupnn.m'));

vgg_path = fullfile(VGG_DIR, 'imagenet-vgg-verydeep-19.mat');

% Specify which layer to use
% "42"/"40" for the layer fc7/fc6 of vgg19, 
% note that the index of layer relu5_4 in vgg19 is 36 but res(i+1).x is the output of layer i. 
opt = struct('layer', 40, 'feat_type', 'vgg19_fc6_matconvnet_resize', 'resize', true); 

net = load(vgg_path);
% Remove FC layers and transfer to GPU
net.layers(opt.layer+1:end)=[];
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu');

cl = numel(classes);
clname = classes{cl};
save_path = fullfile(root, clname, 'features/image/', opt.feat_type);
mkdir(save_path);
imdb = load(fullfile(root, clname, [clname, '.mat']));
averageImage = net.meta.normalization.averageImage;
n = size(imdb.images, 1);
for i = 1:n
  if mod(i,100) == 1
    fprintf('%d ', i);
  end
  % Read raw image
  im = imdb.images{i};
  im_ = single(im);
  if opt.resize
    im_ = imresize(im_, [224,224]);
  end
  if min(size(im, 1), size(im, 2))<224
    rate = 224/min(size(im, 1), size(im, 2));
    im_ = imresize(im_, [rate*size(im, 1), rate*size(im, 2)]);
  end
  [h,w,c] = size(im_);
  % Subtract averageImage
  if  c > 2
      im_ = im_ - imresize(averageImage,[h,w]) ;
  else    
      im_ = bsxfun(@minus,im_,imresize(averageImage,[h,w])) ;
  end

  res = vl_simplenn(net, gpuArray(im_));
  feat = struct;
  feat.data = permute(gather(res(opt.layer).x), [3,1,2]);
  save(fullfile(save_path, sprintf('%d.mat', i)), '-struct', 'feat');
end
fprintf('\n');