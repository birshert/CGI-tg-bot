import click
import torch

import dnnlib
import legacy
from torch_utils import misc
from train import init_dataset_kwargs


def blend(path1, path2, cnt):
    training_set_kwargs, _ = init_dataset_kwargs(data='cartoon')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset

    G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    G_kwargs.channel_base = 32768
    G_kwargs.channel_max = 512
    G_kwargs.mapping_kwargs.num_layers = 2
    G_kwargs.class_name = 'training.networks_stylegan3.Generator'
    G_kwargs.magnitude_ema_beta = 0.5 ** (64 / (20 * 1e3))
    G_kwargs.conv_kernel = 1  # Use 1x1 convolutions.
    G_kwargs.channel_base *= 2  # Double the number of feature maps.
    G_kwargs.channel_max *= 2
    G_kwargs.use_radial_filters = True

    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution,
                         img_channels=training_set.num_channels)

    G1 = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).requires_grad_(False)
    G2 = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).requires_grad_(False)

    with dnnlib.util.open_url(path1) as f:
        resume_data = legacy.load_network_pkl(f)
    misc.copy_params_and_buffers(resume_data['G_ema'], G1, require_all=False)

    with dnnlib.util.open_url(path2) as f:
        resume_data = legacy.load_network_pkl(f)
    misc.copy_params_and_buffers(resume_data['G_ema'], G2, require_all=False)

    state_dict1 = G1.synthesis.state_dict()
    state_dict2 = G2.synthesis.state_dict()

    blend = [0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.5, 0.7, 0.8, .8, .8, .8, 1]

    for key in state_dict2:
        if key[:1] != 'L':
            continue
        l = blend[int(key.split('_')[0][1:])]
        if 'affine' in key:
            l = 0

        state_dict1[key] = state_dict1[key] * l + state_dict2[key] * (1 - l)

    G1.synthesis.load_state_dict(state_dict1)

    return G1


@click.command()
@click.option('--path1', help='Path to model faces', metavar='DIR', required=True)
@click.option('--path2', help='Path to model cartoon', metavar='DIR', required=True)
@click.option('--cnt', help='Num layers', metavar='DIR', required=True, type=int)
@click.option('--path3', help='Output path', metavar='DIR', required=True)
def main(**kwargs):
    path1 = kwargs.get('path1', 'tmp')
    path2 = kwargs.get('path2', 'tmp')
    cnt = kwargs.get('cnt', 5)
    output_path = kwargs.get('path3', 'tmp')

    blended_model = blend(path1, path2, cnt)

    torch.save(blended_model.state_dict(), output_path)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
