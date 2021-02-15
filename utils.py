import os
import platform
import re
from argparse import ArgumentParser
from logger import Logger

import torch

from net import backbones


def map_to_device(device, t):
    """
    Simple helper function to move to a device multiple tensors or networks at once
    :param device:      target device
    :param t:           iterable containing tensors / networks
    :return:            tuple of moved tensors / networks
    """
    return tuple(map(lambda x: x.to(device), t))


def get_default_dataset_root():
    node = platform.node().lower()

    # node = get_cluster_name(node) or node

    known_systems = {
        # 'my-hostname': '/my/data/path'
    }

    path = known_systems.get(node, 'data')
    print("Loading data from {}".format(path))
    return path


def print_args(args, fields):
    """
    Print recap of a selection of args
    :param args:
    :param fields:
    :return:
    """
    for f in fields:
        print(("{:>" + str(1 + max(map(len, fields))) + "} : {:<25}").format(f, getattr(args, f)))


def dict_shortened_summary(config: dict) -> str:
    pairs = list(config.items())
    pairs.sort(key=lambda p: p[0])

    def val2str(val):
        if isinstance(val, float):
            return '{:.3f}'.format(val)
        else:
            return str(val)

    return '_'.join(map(lambda p: '{}{}'.format(''.join(map(lambda s: s[0] + (s[1] if len(s) == 2 else ''),
                                                            re.split(r'[.\-_\s]+', p[0]))), val2str(p[1])), pairs))


def parse_cli_dict(txt: str) -> dict:
    """
    Parse a string like
    var1=abc,var2=4,var3=false
    into a dict.
    Keys without values are interpreted as "True":
    bs=64,use_bottleneck,lr=0.001 is read as
    {bs: 64, use_bottleneck: True, lr: 0.001}
    :param txt:
    :return:
    """
    if txt == '' or str is None:
        return {}

    def parse_val(v: str):
        if type(v) != str:
            return v
        if re.match(r'^[+-]?\d+$', v):
            return int(v)
        if re.match(r'^[+-]?\d+(?:\.\d*)?(?:e[+-]?\d+)?$', v, re.IGNORECASE):
            return float(v)
        if re.match('^y(?:es)?|t|(?:rue)|1|on$', v, re.IGNORECASE):
            return True
        if re.match('^n(?:o)?|f|(?:alse)|0|off$', v, re.IGNORECASE):
            return False
        return v

    return {k: parse_val(v) for k, v in
            map(lambda p: p if len(p) == 2 else (p[0], str(True)), map(lambda p: p.split('='), txt.split(',')))}


def split_dict(config: dict):
    """
    {'disc.num_layers': 3, 'disc.dropout': 0.5, 'net.bottleneck_size': 512}
    to
    {'disc': {'num_layers': 3, 'dropout': 0.5}, 'net': {'bottleneck_size': 512}}
    :param config:
    :return:
    """
    result = {}
    for k in config:
        path = k.split('.')
        parent = path[0]
        value = config[k]
        if len(path) == 1:
            result[parent] = value
        else:
            child = path[1]
            tmp = result.get(parent, {})
            tmp.update({child: value})
            result[parent] = tmp
    return result


def remove_log_hps(config: dict):
    keys = list(config.keys())
    for k in keys:
        if k.endswith('_log'):
            no_log_k = re.sub(r'_log$', '', k)
            if no_log_k in config:
                config.pop(k)
            else:
                config[no_log_k] = 2 ** config.pop(k)


def remove_file(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def add_base_args(parser: ArgumentParser):
    """
    Add to an ArgumentParser the arguments needed for all our settings (source, target, etc)
    :param parser:
    :return:
    """

    # Datasets
    known_datasets = (
        # Office-31
        'amazon', 'dslr', 'webcam',
        # PACS
        'art-pacs', 'cartoon', 'photo', 'sketch-pacs',
        # OfficeHome
        'art-oh', 'clipart-oh', 'realworld', 'product'
    )

    parser.add_argument('--source', type=str, default='amazon', choices=known_datasets, help="Source domain / dataset")
    parser.add_argument('--target', type=str, default='webcam', choices=known_datasets, help="Target domain / dataset")
    parser.add_argument('--data-root', type=str, default=get_default_dataset_root() or 'data', help="Dataset root")

    parser.add_argument('--net', type=str, default='resnet50', choices=backbones, help="Backbone")
    parser.add_argument('--bs', type=int, default=36, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--wd', type=float, default=0.0005, help="Weight decay")
    parser.add_argument('--da', type=str, default='alda', choices=('so', 'dann', 'alda'),
                        help="Domain Adaptation method. SO is Source Only (none)")
    parser.add_argument('--config', type=parse_cli_dict, default='', help="Manual config")

    parser.add_argument('--gpu', type=int, default=0, help="CUDA device to be used")
    parser.add_argument('--load-workers', type=int, default=4, help="Load workers")
    parser.add_argument('--logdir', type=str, default='experiments', help="Log root")

    parser.add_argument('--max-iter', type=int, default=10000, help="Full training length")
    parser.add_argument('--test-iter', type=int, default=100, help="Test interval")
    parser.add_argument('--no-test-source', action='store_true', help="Don't test on source")
    parser.add_argument('--kill-diverging', action='store_true')

    parser.add_argument('--no-tqdm', action='store_true', help="Do not show progress bar")


def add_scalars(writer: Logger, dictionary: dict, global_step: int, prefix='', suffix=''):
    for k in dictionary:
        label = f'{prefix}{k}{suffix}'
        writer.add_scalar(label, dictionary[k], global_step=global_step)
