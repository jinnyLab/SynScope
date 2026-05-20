import argparse

def parse_args():
    desc = "Official Tensorflow 2.5 implementation of ISCL by Kanggeun Lee"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iter', type=int, default=400, help='The number of iters to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch per gpu')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--clip_limit', type=float, default=1.5, help='The clip limit for contrast enhancement')
    parser.add_argument('--training', type=str2bool, default=True, help='Training or deploy')
    parser.add_argument('--data', type=str, help='Directory name to load the training clean data')
    parser.add_argument('--noisy_slide', nargs='+', type=int, help='List of the indices for noisy data (from 0)')
    parser.add_argument('--clean_slide', nargs='+', type=int , help='List of the indices for clean data (from 0)')
    parser.add_argument('--target_range', nargs='+', type=int, help='Range of the targets (start, end)')
    parser.add_argument('--ref_slide', type=int, help='A reference slide for contrast enhancement')
    parser.add_argument('--result_dir', type=str, help='Directory name to save the checkpoints')
    return check_args(parser.parse_args())

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_args(args):
    # --result_dir
    try:
        assert args.epoch >= 1
    except:
        print('The number of epochs must be larger than or equal to one')

    # --batch_size
    assert args.batch_size >= 1, ('Batch size must be larger than or equal to one')
    try:
        os.mkdir(args.result_dir)
    except:
        print('Directory already exists or cannot make')

    return args

