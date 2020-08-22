args = struct;

%--------------------------------------------------------------------------------------

small_imdb = false;
save_result = true;

[args.root{all_classes}] = deal(fullfile(DATA_ROOT, imgset));
args.clname = get_classes(imgset);
[args.small_imdb{all_classes}] = deal(small_imdb);
[args.feat_type{all_classes}] = deal(feat_type);
[args.save_root{all_classes}] = deal(fullfile(DATA_ROOT, save_dir));
[args.save_result{all_classes}] = deal(save_result);

%--------------------------------------------------------------------------------------

alpha = 0.3;
num_maxima = 20;
ws = 3;
conn = 4;
beta_global = 0.5;
beta_local = 1;
num_levels = 50;

[args.alpha{all_classes}] = deal(alpha);
[args.num_maxima{all_classes}] = deal(num_maxima);
[args.ws{all_classes}] = deal(ws);
[args.conn{all_classes}] = deal(conn);
[args.beta_global{all_classes}] = deal(beta_global);
[args.beta_local{all_classes}] = deal(beta_local);
[args.num_levels{all_classes}] = deal(num_levels);

%----------------------------------------------------------------------------------------

[proposals, root_ids, root_feat_types] = create_proposals_cnn(args, class_indices, row_indices);
