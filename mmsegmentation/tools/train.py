# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

    # classes ThyroidUS
    # classes = ('internal background',
    #         'connective tissue',
    #         'trachea',
    #         'thyroid gland',
    #         'blood vessels',
    #         'dermal',
    #         'muscle tissue',
    #         'thyroid nodule',
    #         'external background')

    # # classes IMT
    # classes = ('lumen',
    #         'far wall',
    #         'near wall',
    #         'hyper echo',
    #         'low echo',
    #         'other',
    #         'background')

    # classes DTP
    # classes = ('background',
    #         'cell')

    # # classes VERSE
    # classes = ('background',
    #         'vertebrae')
    
    # # classes RETINA
    # classes = ('background',
    #         'vessel')
    
    # # classes FSHD
    # classes = ('background',
    #         'Biceps_brachii', # 001 - 1 for label
    #         'Deltoideus', # 002
    #         'Depressor_anguli_oris', # 003
    #         'Digastricus', # 004
    #         'Gastrocnemius_medial_head', # 008
    #         'Geniohyoideus', # 009
    #         'Masseter', # 011
    #         'Mentalis', # 012
    #         'Orbicularis_oris', # 013
    #         'Rectus_abdominis', # 015
    #         'Rectus_femoris', # 016
    #         'Temporalis', # 017
    #         'Tibialis_anterior', # 018
    #         'Trapezius', # 019
    #         'Vastus_lateralis', # 020
    #         'Zygomaticus')  # 021
    
    # classes FSHD
    classes = ('background',
            'muscle')
    
    # classes DERMA
    # classes = ('background',
    #          'lesion') 
 
    # paletteThyroid = [
    #     (0, 0, 0), # internal background - black
    #     (0, 255, 0), # connective tissue - green
    #     (0, 0, 255), # trachea - blue
    #     (255, 0, 0), # thyroid gland - red
    #     (255, 255, 0), # blood vessels - yellow
    #     (255, 0, 255), # dermal - magenta
    #     (0, 255, 255), # muscle tissue - cyan
    #     (255, 255, 255), # thyroid nodule - white
    #     (128, 128, 128) # external background - gray
    # ]

    # paletteIMT = [
    #     (0, 0, 0), # lumen - black
    #     (0, 255, 0), # far wall - green
    #     (0, 0, 255), # near wall - blue
    #     (255, 0, 0), # hyper echo - red
    #     (255, 255, 0), # low echo - yellow
    #     (255, 0, 255), # other - magenta
    #     (0, 255, 255), # background - cyan
    # ]

    # paletteDTP = [
    #     (0, 0, 0), # background - black
    #     (255, 255, 255), # cell - white
    # ]

    # paletteVERSE = [
    #     (0, 0, 0), # background - black
    #     (255, 255, 255), # vertebrae - white
    # ]
    
    # paletteRETINA = [
    #     (0, 0, 0), # background - black
    #     (255, 255, 255), # vessel - white
    # ]

    # paletteFSHD = [
    #         (0, 0, 0),       # black
    #     (128, 0, 128),   # purple
    #     (0, 128, 128),   # teal
    #     (128, 128, 128), # gray
    #     (255, 0, 0),     # red
    #     (0, 255, 0),     # lime
    #     (255, 255, 0),   # yellow
    #     (0, 0, 255),     # blue
    #     (255, 0, 255),   # fuchsia
    #     (0, 255, 255),   # aqua
    #     (192, 192, 192), # silver
    #     (255, 255, 255), # white
    #     (255, 99, 71),   # tomato
    #     (255, 69, 0),    # orange-red
    #     (255, 165, 0),   # orange
    #     (255, 215, 0),   # gold
    #     (46, 139, 87)]   # sea green


    paletteFSHD = [
            (0, 0, 0),       # black
        (128, 0, 128),   # purple
        ]  
    
    # paletteDERMA = [
    #     (0, 0, 0), # background - black
    #     (128, 0, 128), # lesion - white
    # ]
    
    # @DATASETS.register_module()
    # class ThyroidUS(BaseSegDataset):
    #     METAINFO = dict(classes = classes, palette = paletteThyroid)
    #     def __init__(self, **kwargs):
    #         super().__init__(img_suffix='.png',
    #                         seg_map_suffix='.png',
    #                         reduce_zero_label = False,
    #                         **kwargs)
            
    # @DATASETS.register_module()
    # class IMT(BaseSegDataset):
    #     METAINFO = dict(classes = classes, palette = paletteIMT)
    #     def __init__(self, **kwargs):
    #         super().__init__(img_suffix='.png',
    #                         seg_map_suffix='.png',
    #                         reduce_zero_label = False,
    #                         **kwargs)
            
    # @DATASETS.register_module()
    # class DTP(BaseSegDataset):
    #     METAINFO = dict(classes = classes, palette = paletteDTP)
    #     def __init__(self, **kwargs):
    #         super().__init__(img_suffix='.png',
    #                         seg_map_suffix='.png',
    #                         reduce_zero_label = False,
    #                         **kwargs)
    
    # @DATASETS.register_module()
    # class VERSE(BaseSegDataset):
    #     METAINFO = dict(classes = classes, palette = paletteVERSE)
    #     def __init__(self, **kwargs):
    #         super().__init__(img_suffix='.png',
    #                         seg_map_suffix='.png',
    #                         reduce_zero_label = False,
    #                         **kwargs)
    
    # # @DATASETS.register_module()
    # class RETINA(BaseSegDataset):
    #     METAINFO = dict(classes = classes, palette = paletteRETINA)
    #     def __init__(self, **kwargs):
    #         super().__init__(img_suffix='.png',
    #                         seg_map_suffix='.png',
    #                         reduce_zero_label = False,
    #                         **kwargs)
            
    @DATASETS.register_module()
    class FSHD(BaseSegDataset):
        METAINFO = dict(classes = classes, palette = paletteFSHD)
        def __init__(self, **kwargs):
            super().__init__(img_suffix='.png',
                            seg_map_suffix='.png',
                            reduce_zero_label = False,
                            ignore_index=255,
                            **kwargs)

    # @DATASETS.register_module()
    # class DERMA(BaseSegDataset):
    #     METAINFO = dict(classes = classes, palette = paletteDERMA)
    #     def __init__(self, **kwargs):
    #         super().__init__(img_suffix='.png',
    #                         seg_map_suffix='.png',
    #                         reduce_zero_label = False,
    #                         **kwargs)
            
    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
