import argparse
import os
from .compress.compress import BagCompress as bc
from .process.process import BagProcess as bp

def process(args):

    if args.batch is not None:
        bp.process_batch(bag_folder_path=args.batch,
                        dst_dataset=args.dst,
                        dst_datafolder_name=args.name,
                        save_raw=args.no_raw,
                        force_process=args.force,
                        mode = args.mode)
        
    elif args.folder is not None:
        bp.process_folder(folder_path=args.folder,
                        dst_dataset=args.dst,
                        dst_datafolder_name=args.name,
                        save_raw=args.no_raw,
                        force_process=args.force,
                        mode = args.mode)

def compress(args):

    if args.batch is not None:
        bc.compress_batch(args.batch)

    elif args.folder is not None:
        bc.compress_folder(args.folder)
        

def reset(args):
    if args.batch is not None:
        bp.reset_batch(args.batch)

    elif args.folder is not None:
        bp.reset_folder(args.folder)

def is_bag_file(arg_bag_str: str) -> str:
    """"""
    # check file validation
    if os.path.isfile(arg_bag_str) and arg_bag_str.split('.')[-1]=='bag':
        return arg_bag_str
    else:
        msg = f"Given bag file {arg_bag_str} is not valid! "
        raise argparse.ArgumentTypeError(msg)

def is_bag_dir(arg_bag_str:str):
    # check dir validation
    if os.path.isdir(arg_bag_str):
        return arg_bag_str
    else:
        msg = f"Given bag directory {arg_bag_str} is not valid! "
        raise argparse.ArgumentTypeError(msg)

def main():
    # Create the top-level parser
    parser = argparse.ArgumentParser(prog='bagtool')
    subparsers = parser.add_subparsers(help='sub-command help')

    # Create the parser for the "compress" command
    parser_compress = subparsers.add_parser('compress',
                                        help='compress help')
    parser_compress.add_argument('-b', '--batch', type=is_bag_dir,
                            help='path to a bag batch folder')
    parser_compress.add_argument('-f', '--folder', type=is_bag_dir,
                            help='path to a bag folder consisting bag batches')
    parser_compress.set_defaults(func=compress)
    
    # Create the parser for the "process" command
    parser_process = subparsers.add_parser('process',
                                        help='process help')
    parser_process.add_argument('-b', '--batch', type=is_bag_dir,
                            help='path to a bag batch folder')
    parser_process.add_argument('-f', '--folder', type=is_bag_dir,
                            help='path to a bag folder consisting bag batches')
    parser_process.add_argument('-d', '--dst',default=None,
                            help='path to a dataset destination folder')
    parser_process.add_argument('-n', '--name',default=None,
                            help='a custom name for the datafolder')
    parser_process.add_argument('-m', '--mode',default="data",
                            help='a custom name for the datafolder')
    parser_process.add_argument('--no_raw',action="store_false",
                            help='not saving raw data')
    parser_process.add_argument('--force',action="store_true",
                            help='not saving raw data')
    parser_process.set_defaults(func=process)


    # Create the parser for the "process" command
    parser_reset = subparsers.add_parser('reset',
                                        help='reset help')
    parser_reset.add_argument('-b', '--batch', type=is_bag_dir,
                            help='path to a bag batch folder')
    parser_reset.add_argument('-f', '--folder', type=is_bag_dir,
                            help='path to a bag folder consisting bag batches')
    parser_reset.set_defaults(func=reset)

    # Parse the args and call the default function
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()