original_imgset = 'vocx';
cnn_imgset = 'vocx_cnn';

% prepare data
fprintf('Preparing data ...\n');
cd(fullfile(ROOT, 'commons/utils'));
imgset = original_imgset;
create_lite_imdb;

% extract image features
fprintf('Extracting image features ...\n');
cd(fullfile(ROOT, 'commons/deepfeat'));
addpath(genpath(fullfile(ROOT, 'commons/utils')));
imgset = original_imgset;
get_feat_image_matconvnet_relu;
get_feat_image_matconvnet_fc;
compute_neighbors_from_fc_layers;

% create proposals
fprintf('Generating proposals ...\n');
cd(fullfile(ROOT, 'commons/proposals'));
addpath(fullfile(ROOT, 'commons/utils'));
addpath(fullfile(ROOT, 'commons/metrics'));
imgset = original_imgset;
classes = get_classes(imgset);
all_classes = 1:numel(classes);
save_dir = cnn_imgset;
imdb = load(fullfile(DATA_ROOT, imgset, 'mixed/mixed_lite.mat'), 'bboxes');
class_indices = numel(classes);
row_indices = 1:numel(imdb.bboxes);

feat_type = 'vgg19_relu44_matconvnet';
script;
feat_type = 'vgg19_relu54_matconvnet';
script;

fprintf('Compressing and combining proposals ...\n');
imgset = cnn_imgset;
classes = get_classes(imgset);
proposal_name = 'vgg19_relu44_matconvnet_lite';
compress_proposals;
proposal_name = 'vgg19_relu54_matconvnet_lite';
compress_proposals;

combine_proposals;


% extract region features
fprintf('Extracting region features ...\n');
cd(fullfile(ROOT, 'commons/deepfeat'));
imgset = cnn_imgset;
images_imgset = original_imgset;
get_feat_roipool_matconvnet;


% create class data from mixed
fprintf('Create class imdb from mixed imdb ...\n');
imgset = cnn_imgset;
bboxes_imgset = original_imgset;
create_classes_imdb_from_mixed_cnn;

feat_type = 'features/proposals/vgg19_relu54_matconvnet_77_roi_pooling_noresize';
copy_class_features_from_mixed;