all_classes = 1:21;
big_classes = [1 3 7 8 9 12 15 21];
SAVE_RESULT = true;
%--------------------------------------------------------------------------------------

imgset = 'voc12_traival_vgg19_matconvnet_cnn_03_20_1_05';
small_imdb = false;
lite_imdb = true;
nu = 50;
tau = 10;
nms_IoU = 0.3;

[args.root{all_classes}] = deal(fullfile('~/', imgset));
args.clname = get_classes(imgset);
[args.small_imdb{all_classes}] = deal(small_imdb);
[args.lite_imdb{all_classes}] = deal(lite_imdb);
[args.nu{all_classes}] = deal(nu);
[args.tau{all_classes}] = deal(tau);
[args.nms_IoU{all_classes}] = deal(nms_IoU);
%--------------------------------------------------------------------------------------

sctype = 'confidence_imdb'; % 'confidence' or 'standout'
scname = 'vgg19_relu54_matconvnet_77_roi_pooling_noresize_01_symmetric';
normsc = true;
rounding = '50';
iterating = '50';
data_loader = @DataLoader;

[args.sctype{all_classes}] = deal(sctype); % 'confidence' or 'standout'
[args.scname{all_classes}] = deal(scname);
[args.normsc{all_classes}] = deal(normsc);
[args.rounding{all_classes}] = deal(rounding);
[args.iterating{all_classes}] = deal(iterating);
[args.data_loader{all_classes}] = deal(data_loader);

scname_nb = 'vgg19_relu54_matconvnet_77_roi_pooling_noresize_01_symmetric_50_neighbor_cos_vgg19_fc6_matconvnet_resize';
[args.scname{big_classes}] = deal(scname_nb);

%--------------------------------------------------------------------------------------

[args.prefiltered_nb{all_classes}] = deal(false);
[args.num_nb_rounding{all_classes}] = deal(0);
[args.num_nb_iterating{all_classes}] = deal(0);
[args.nb_root{all_classes}] = deal({});
[args.nb_type{all_classes}] = deal({''});

num_nb_rounding = 50;
num_nb_iterating = 50;
nb_root = {...
           '~/voc12_trainval', ...
          };
nb_type = {...
           'neighbor_cos_vgg19_fc6_matconvnet_resize', ...
          };
[args.prefiltered_nb{big_classes}] = deal(true);
[args.num_nb_rounding{big_classes}] = deal(num_nb_rounding);
[args.num_nb_iterating{big_classes}] = deal(num_nb_iterating);
[args.nb_root{big_classes}] = deal(nb_root);
[args.nb_type{big_classes}] = deal(nb_type);

%----------------------------------------------------------------------------------------

save_result = SAVE_RESULT;
vis_result = false;
vis_root = '';

[args.save_result{all_classes}] = deal(save_result);
[args.vis_result{all_classes}] = deal(vis_result);
[args.vis_root{all_classes}] = deal(vis_root);

%----------------------------------------------------------------------------------------

[args.optim_sol{all_classes}] = deal('one');
[args.res_path{all_classes}] = deal('');
%----------------------------------------------------------------------------------------
[args.verbose{all_classes}] = deal(true);
%----------------------------------------------------------------------------------------
[cl1, cl5, cln, sol] = compute_solutions(args, class_indices, solution_indices);
