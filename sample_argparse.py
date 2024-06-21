import argparse

parser = argparse.ArgumentParser()

# MODEL PARAMS
parser.add_argument('--model1', type=str, default='vit_small_patch16_224.augreg_in21k_ft_in1k')
parser.add_argument('--model1_ckpt', type=str, default=None)
parser.add_argument('--model2', type=str, default='vit_large_patch16_224.augreg_in21k_ft_in1k')
parser.add_argument('--model2_ckpt', type=str, default=None)
parser.add_argument('--split1', type=str, default='blocks_11')
parser.add_argument('--split2', type=str, default='blocks_11')

# DATASET PARAMS
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--dataset_split', type=str, default='train')
parser.add_argument('--num_images', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--class_list_path', type=str, default='class_lists/class_list_vit_test1.json')

# CRAFT PARAMS
parser.add_argument('--num_concepts', type=int, default=10)
parser.add_argument('--patch_size', type=int, default=224)

# MODIFIED CRAFT PARAMS
parser.add_argument('--image_group_strategy', type=str, default='union')
parser.add_argument('--normalize_w', type=bool, default=False)
parser.add_argument('--normalize_a', type=bool, default=False)
parser.add_argument('--norm_method', type=str, default=None)
parser.add_argument('--viz_sim_matrices', type=bool, default=True)
parser.add_argument('--viz_crop_concept_scores', type=bool, default=True)

# CKA PARAMS
parser.add_argument('--kernel', type=str, default='linear')
parser.add_argument('--no_debiased', dest='debiased', action='store_false')

args = parser.parse_args()

print(args)