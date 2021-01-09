"""Handles downloading pre-trained model checkpoints and datasets."""

import wget
import os


def download(path, local_path):
    url = f'http://efrosgans.eecs.berkeley.edu/HessianPenalty/resources/{path}'
    print(url)
    wget.download(url, local_path)


def find_dataset(name):
    valid_datasets = ['clevr_simple', 'clevr_complex', 'clevr_1fov', 'clevr_u', 'edges_and_shoes']
    if name not in valid_datasets:
        return False
    max_tfr_lod = 8 if name != 'edges_and_shoes' else 7
    base_dest = os.path.join('datasets', name)
    if not os.path.exists(base_dest):
        print(f'Downloading f{name} dataset...')
        os.makedirs(f'datasets/{name}', exist_ok=True)
        for i in range(2, max_tfr_lod + 1):
            url_path = os.path.join(base_dest, '%s-r%02d.tfrecords' % (name, i))
            print(url_path)
            download(url_path, url_path)
    return os.path.join(os.getcwd(), base_dest)


def find_model(name):
    valid_prefix = ['clevr_simple', 'clevr_complex', 'clevr_1fov', 'clevr_u', 'edges_and_shoes']
    valid_suffix = ['_ft', '_fs', '_info', '_bl']
    dset_map = {'simple': 'clevr_simple',
                'complex': 'clevr_complex',
                '1fov': 'clevr_1fov',
                'underparam': 'clevr_u',
                'e2s': 'edgeshoes'}
    full_name_to_abbrev = {'clevr_simple': 'simple',
                           'clevr_complex': 'complex',
                           'clevr_1fov': '1fov',
                           'clevr_u': 'underparam',
                           'edges_and_shoes': 'e2s'}
    legal = any([name.startswith(prefix) for prefix in valid_prefix]) and any([name.endswith(suffix) for suffix in valid_suffix])
    if not legal:
        return False
    name_comps = name.split("_")
    abbrev_name = f'{full_name_to_abbrev["_".join(name_comps[:-1])]}_{name_comps[-1]}'
    # Files on servers use an abbreviated naming format:
    dest = os.path.join('pretrained_models', dset_map[abbrev_name.split('_')[0]], f'{abbrev_name}.pkl')
    local_dest = os.path.join('pretrained_models', f'{name}.pkl')
    if not os.path.exists(local_dest):
        print(f'Downloading pre-trained model {name}...')
        download(dest, local_dest)
    return local_dest


find_model('simple_fs')
