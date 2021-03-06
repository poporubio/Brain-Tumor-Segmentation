import argparse


def brain_tumor_argparse():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Experiment')
    parser.add_argument(
        '-m',
        '--model_id',
        type=str,
        help='The model_id to be used.',
    )
    parser.add_argument(
        '-d',
        '--data_provider_id',
        type=str,
        help='The medical-image data provider.',
    )
    parser.add_argument(
        '-grs',
        '--global_random_seed',
        type=int,
        help='The global random seed. [5566]',
        default=5566,
    )
    parser.add_argument(
        '--comet',
        dest='do_comet',
        action='store_true',
        help='Use comet-ml to document the info.',
    )
    parser.add_argument(
        '-cop',
        '--comet_project',
        type=str,
        help='comet ml project name',
        default='braintumorbaba',
    )
    parser.add_argument(
        '-cow',
        '--comet_workspace',
        type=str,
        help='comet ml workspace name',
        default='raywu0123',
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='checkpoint directory to load the model',
    )
    parser.set_defaults(do_comet=False)
    return parser
