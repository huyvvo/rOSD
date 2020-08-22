root = fullfile(DATA_ROOT, imgset);
classes = get_classes(imgset);

% Run matconvnet setup, please change the location as your own
run(fullfile(MATCONVNET_PATH, 'matlab/vl_setupnn.m'));

vgg_path = fullfile(VGG_DIR, 'imagenet-vgg-verydeep-19.mat');

configs = {{28, 'vgg19_relu44_matconvnet'}, {37, 'vgg19_relu54_matconvnet'}};

cl = numel(classes);
clname = classes{cl};
imdb = load(fullfile(root, clname, [clname, '.mat']));
n = size(imdb.images, 1);

for conf_idx = 1:numel(configs)
  conf = configs{conf_idx};
  % Specify which layer to use
  % "37"/"35"/28"/"19" for the layer relu5_4/relu5_3/relu4_4/relu3_4 of vgg19, 
  % note that the index of layer relu5_4 in vgg19 is 36 but res(i+1).x is the output of layer i. 
  opt = struct('layer', conf{1}, 'feat_type', conf{2}); 
  net = load(vgg_path);
  % Remove FC layers and transfer to GPU
  net.layers(opt.layer+1:end)=[];
  net = vl_simplenn_tidy(net);
  net = vl_simplenn_move(net, 'gpu');
  
  save_path = fullfile(root, clname, 'features/image/', opt.feat_type);
  mkdir(save_path);
  
  averageImage = net.meta.normalization.averageImage;
  for i = 1:n
    if mod(i,100) == 1
      fprintf('%d ', i);
    end
    % Read raw image
    im = imdb.images{i};
    im_ = single(im);
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
end
fprintf('\n');