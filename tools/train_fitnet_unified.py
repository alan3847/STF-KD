# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')
import inspect
from openstl.api import BaseExperiment
from openstl.utils.parser import create_parser, default_parser
from openstl.utils.main_utils import get_dist_info, load_config, update_config


def resolve_student_cfg_path(args):
    """Resolve which config file should be used as the primary/student config."""
    # Priority:
    # 1) --student_config_file
    # 2) --config_file
    # 3) fallback to ./configs/{dataname}/{student_method}.py
    # 4) if student_method is simvp subtype and user wants subtype config,
    #    they should pass --student_config_file explicitly

    if args.student_config_file is not None:
        return args.student_config_file

    if args.config_file is not None:
        return args.config_file

    if args.student_method is None:
        raise ValueError(
            "For KD training, `student_method` must be provided, and a student config "
            "should be provided via `--student_config_file` or `--config_file` "
            "unless a plain fallback config exists."
        )

    return osp.join('./configs', args.dataname, f'{args.student_method}.py')


if __name__ == '__main__':
    print("DEBUG train_kd_unified file:", __file__)
    print("DEBUG create_parser source:", inspect.getsourcefile(create_parser))
    p = create_parser()
    print("DEBUG parser args:", sorted([a.dest for a in p._actions]))
    args = p.parse_args()

    # --------------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------------
    if args.teacher_method is None:
        raise ValueError("`--teacher_method` must be provided for KD training.")
    if args.student_method is None:
        raise ValueError("`--student_method` must be provided for KD training.")
    if args.teacher_ckpt is None and not args.test:
        print("[train_kd_unified] Warning: `--teacher_ckpt` is not provided. "
              "Teacher will be randomly initialized, which is usually not what you want.")

    # Force method to KD
    args.method = 'fitnet_kd'

    # Normalize method names
    args.teacher_method = args.teacher_method.lower()
    args.student_method = args.student_method.lower()

    # --------------------------------------------------------------
    # Load primary config (student config)
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # Re-assert KD-specific keys after config merge
    # (avoid being overwritten by student config file)
    # --------------------------------------------------------------
    config['method'] = 'fitnet_kd'
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


    # --------------------------------------------------------------
    # Important consistency fixes
    # --------------------------------------------------------------
    # student config may have method='SimVP' / 'PhyDNet' / ...
    # but experiment should build KDMethod
    args.method = 'fitnet_kd'

    # ensure args namespace reflects merged config
    for k, v in config.items():
        setattr(args, k, v)

    print('>' * 35 + ' training ' + '<' * 35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>' * 35 + ' testing  ' + '<' * 35)
    mse = exp.test()