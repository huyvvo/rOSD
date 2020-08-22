args = struct;

%--------------------------------------------------------------------------------------

small_imdb = false;
saveconf = true;
savestd = false;

[args.root{all_classes}] = deal(fullfile(DATA_ROOT, imgset));
args.clname = get_classes(imgset);
[args.feat_root{all_classes}] = deal(fullfile(DATA_ROOT, imgset));
[args.small_imdb{all_classes}] = deal(small_imdb);
[args.saveconf{all_classes}] = deal(saveconf);
[args.savestd{all_classes}] = deal(savestd);

%-------------------------------
PHM_type ='';
symmetric = true;
num_pos = 1000;
num_pos_text = '1000';
stdver = 4;
max_iter = 10000;
area_ratio = 0.5;
area_ratio_text = '05';

[args.PHM_type{all_classes}] = deal(PHM_type);
[args.symmetric{all_classes}] = deal(symmetric);
[args.num_pos{all_classes}] = deal(num_pos);
[args.num_pos_text{all_classes}] = deal(num_pos_text);
[args.stdver{all_classes}] = deal(stdver);
[args.max_iter{all_classes}] = deal(max_iter);
[args.area_ratio{all_classes}] = deal(area_ratio);
[args.area_ratio_text{all_classes}] = deal(area_ratio_text);

%-------------------------------
deepfeat = true;
feat_type = 'vgg19_relu54_matconvnet_77_roi_pooling_noresize';
sim_type = '01';

[args.deepfeat{all_classes}] = deal(deepfeat);
[args.feat_type{all_classes}] = deal(feat_type);
[args.sim_type{all_classes}] = deal(sim_type);

%-------------------------------

[args.prefiltered_nb{all_classes}] = deal(false);
[args.num_nb{all_classes}] = deal(0);
[args.nb_root{all_classes}] = deal({});
[args.nb_type{all_classes}] = deal({''});

prefiltered_nb = true;
num_nb = 50;
nb_root = {...
           fullfile(DATA_ROOT, neighbor_imgset), ...
          };
nb_type = {...
           'neighbor_cos_vgg19_fc6_matconvnet_resize', ...
          };
[args.prefiltered_nb{big_classes}] = deal(prefiltered_nb);
[args.num_nb{big_classes}] = deal(num_nb);
[args.nb_root{big_classes}] = deal(nb_root);
[args.nb_type{big_classes}] = deal(nb_type);

%-------------------------------
compute_scores(args, class_indices, row_indices);