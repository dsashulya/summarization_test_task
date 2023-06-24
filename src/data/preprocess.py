import argparse
import os

from typing import List, NoReturn, Tuple

from data import read_data, write_data


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data',
                        type=str,
                        required=True)
    parser.add_argument('--source_fname',
                        type=str,
                        default='reddit.val.source.remove_markers_simple_separator.txt',
                        required=False)
    parser.add_argument('--target_fname',
                        type=str,
                        default='reddit.val.target.remove_markers_simple_separator.txt',
                        required=False)
    parser.add_argument('--reference_fname',
                        type=str,
                        default='val.target',
                        required=False,
                        help='Original validation target data to be used as index reference for the split')

    return parser


def train_val_split(source: List[str],
                    target: List[str],
                    reference_val_target: List[str]
                    ) -> Tuple[List[str],
                               List[str],
                             List[str],
                             List[str]]:
    reference_target_set = set(reference_val_target)
    val_idx = [i for i, item in enumerate(target) if item in reference_target_set]
    train_idx = [i for i in range(len(source)) if i not in set(val_idx)]
    return [source[i] for i in train_idx], \
            [source[i] for i in val_idx], \
            [target[i] for i in train_idx], \
            [target[i] for i in val_idx]


def main(args) -> NoReturn:
    source = read_data(args.path_to_data, args.source_fname)
    target = read_data(args.path_to_data, args.target_fname)
    reference = read_data(args.path_to_data, args.reference_fname)

    X_train, X_val, y_train, y_val = train_val_split(source,
                                                     target,
                                                     reference)
    for fname, data in zip(['train.source',
                            'val.source',
                            'train.target',
                            'val.target'],
                             [X_train,
                              X_val,
                              y_train,
                              y_val]):
        path = os.path.join(args.ath_to_data, fname)
        write_data(path, data)


if __name__ == '__main__':
    args = setup_argparser().parse_args()
    main(args)
