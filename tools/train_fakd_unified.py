# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')
import inspect

from openstl.api import BaseExperiment
from openstl.utils.parser import create_parser, default_parser
from openstl.utils.main_utils import get_dist_info, load_config, update_config


def resolve_student_cfg_path(args):
    if args.student_config_file is not None:
        return args.student_config_file

    if args.config_file is not None:
        return args.config_file

    if args.student_method is None:
        raise ValueError(
            "For FAKD training, `student_method` must be provided, and a student config "
            "should be provided via `--student_config_file` or `--config_file`."
        )

    return osp.join('./configs', args.dataname, f'{args.student_method}.py')


if __name__ == '__main__':
    print("DEBUG train_fakd_unified file:", __file__)
    print("DEBUG create_parser source:", inspect.getsourcefile(create_parser))

    p = create_parser()
    args = p.parse_args()

    if args.teacher_method is None:
        raise ValueError("`--teacher_method` must be provided for FAKD training.")
    if args.student_method is None:
        raise ValueError("`--student_method` must be provided for FAKD training.")
    if args.teacher_ckpt is None and not args.test:
        print("[train_fakd_unified] Warning: `--teacher_ckpt` is not provided. "
              "Teacher will be randomly initialized, which is usually not what you want.")
    if args.teacher_hint_layer is None:
        raise ValueError("`--teacher_hint_layer` must be provided for FAKD training.")
    if args.student_hint_layer is None:
        raise ValueError("`--student_hint_layer` must be provided for FAKD training.")

    args.method = 'fakd_kd'
    args.teacher_method = args.teacher_method.lower()
    args.student_method = args.student_method.lower()

    config = args.__dict__
    cfg_path = resolve_student_cfg_path(args)

    if args.overwrite:
        config = update_config(
            config,
            load_config(cfg_path),
            exclude_keys=['method']
        )
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(
            config,
            loaded_cfg,
            exclude_keys=[
                'method',
                'val_batch_size',
                'drop_path',
                'warmup_epoch'
            ]
        )
        default_values = default_parser()
        for attribute in default_values.keys():
            if config.get(attribute, None) is None:
                config[attribute] = default_values[attribute]

    config['method'] = 'fakd_kd'
    config['teacher_method'] = args.teacher_method
    config['student_method'] = args.student_method
    config['teacher_ckpt'] = args.teacher_ckpt
    config['student_ckpt'] = args.student_ckpt
    config['kd_weight'] = args.kd_weight
    config['teacher_config_file'] = args.teacher_config_file
    config['student_config_file'] = args.student_config_file if args.student_config_file is not None else cfg_path
    config['config_file'] = cfg_path

    config['teacher_hint_layer'] = args.teacher_hint_layer
    config['student_hint_layer'] = args.student_hint_layer
    config['hint_weight'] = args.hint_weight
    config['hint_teacher_channels'] = args.hint_teacher_channels
    config['hint_student_channels'] = args.hint_student_channels
    config['hint_reduce'] = args.hint_reduce
    config['teacher_hint_output_index'] = args.teacher_hint_output_index
    config['student_hint_output_index'] = args.student_hint_output_index

    config['fakd_weight'] = args.fakd_weight
    config['freq_alpha'] = args.freq_alpha
    config['freq_cutoff'] = args.freq_cutoff
    config['freq_loss_type'] = args.freq_loss_type
    config['output_kd'] = args.output_kd
    config['freq_norm'] = args.freq_norm
    config['freq_eps'] = args.freq_eps
    config['freq_log_mag'] = args.freq_log_mag


    args.method = 'fakd_kd'

    for k, v in config.items():
        setattr(args, k, v)

    print('>' * 35 + ' training ' + '<' * 35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>' * 35 + ' testing  ' + '<' * 35)
    mse = exp.test()
