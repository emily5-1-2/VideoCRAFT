

model_names = {
    'vit_small_patch16_224.augreg_in21k_ft_in1k': 'vit_s',
    'vit_base_patch16_224.augreg_in21k_ft_in1k': 'vit_b',
    'vit_large_patch16_224.augreg_in21k_ft_in1k': 'vit_l',
    'resnet18.a2_in1k': 'r18',
    'resnet34.a2_in1k': 'r34',
    'resnet50.a2_in1k': 'r50',
}

param_value_names = {
    'image_group_strategy': {'union': 'u', 'intersection': 'i'},
    'normalize_w': {True: 'nw', False: None},
    'normalize_a': {True: 'na', False: None},
    'norm_method': {None: None, 'min_max': 'mm', 'z_score': 'zs'},
    'patch_size': lambda x: f'{x}',
    'num_concepts': lambda x: f'{x}',
    'seed': lambda x: f'{x}',
    'split_params': lambda x: convert_split_params(x),
    'match_name': lambda x: convert_model_tuple(x[0]) + '_v_' + convert_model_tuple(x[1]),
    'match_params': lambda x: convert_match_params(x),
    'kernel': {'linear': 'lin', 'rbf': 'rbf'},
    'debiased': {True: 'deb', False: None},
}
param_key_names = {
    'image_group_strategy': 'igs=',
    'normalize_w': '',
    'normalize_a': '',
    'norm_method': 'nm=',
    'patch_size': 'ps=',
    'num_concepts': 'nc=',
    'split_params': '',
    'match_params': '',
    'match_name': '',
    'seed': 'seed=',
    'kernel': 'kern=',
    'debiased': '',
}


def convert_match_params(match_params):
    model1, model2 = match_params
    m1 = convert_model_tuple(model1['model_tuple']) + '_' + convert_split_params(model1['split_params'])
    m2 = convert_model_tuple(model2['model_tuple']) + '_' + convert_split_params(model2['split_params'])
    return f'{m1}_v_{m2}'

def convert_model_tuple(model_tuple):
    model_name, ckpt = model_tuple
    model_abbr = model_names[model_name]

    name = f'{model_abbr}_ckpt={ckpt}'
    return name


def convert_split_params(split_params):
    split_layer = split_params['split_layer']
    split_point = split_params.get('split_point', None)
    if split_point is not None:
        split_point = f'{split_point}'
    else:
        split_point = ''
    name = f'{split_layer}_{split_point}'
    return name


def convert_run_params(run_params):

    name = ''
    for key, val in run_params.items():
        if key in param_value_names:
            kn = param_key_names[key]
            if callable(param_value_names[key]):
                v = param_value_names[key](val)
            else:
                v = param_value_names[key][val]
            if v:
                name = name + kn + v + '_'
        else:
            raise ValueError(f'Unknown key {key}')

    return name[:-1]