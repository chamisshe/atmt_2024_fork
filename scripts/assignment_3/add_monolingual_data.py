#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Use this script to extract deduplicated and shuffled src-tgt pairs from
src-tgt aligned files downloaded from OPUS.

Example call:

    python extract_splits.py --src /mnt/storage/clwork/users/kew/HS2021/infopankki_raw/infopankki.en-sv.sv --tgt /mnt/storage/clwork/users/kew/HS2021/infopankki_raw/infopankki.en-sv.en --outdir /mnt/storage/clwork/users/kew/HS2021/atmt/data/en-sv/infopankki/raw

"""

import argparse
import os
import shutil
import random
from pathlib import Path

def set_args():

    ap = argparse.ArgumentParser()

    ap.add_argument('--in-dir', required=True, default="./data/en-fr/raw", type=str, help='Directory containing the train/test/validation files.')
    ap.add_argument('--src-lang', required=True, type=str, help='Source language.')
    ap.add_argument('--tgt-lang', required=True, type=str, help='Target language.')
    ap.add_argument('--out-dir', required=True, default="./data/en-fr-enhanced/raw", type=str, help='')
    # ap.add_argument('--training_data', required=True, type=str, help='The file including')
    ap.add_argument('--mono-data', required=True, type=str, help='Textfile containing the monolingual training data (in the target language).')
    
    # ap.add_argument('--out-name', required=False, default="train_enhanced", type=str, help='Basename to name the resulting training-data files.')
    ap.add_argument('--mono-size', required=False, default=10000, type=int, help='Amount of lines (=sentences) to use from the monolingual data.')
    # ap.add_argument('--tiny_train_size', required=False, default=1000, type=int, help='')

    ap.add_argument('--shuffle', required=False, action="store_true", help='Whether to shuffle the resulting ("enhanced") datasets.')
    ap.add_argument('--random-sample', required=False, action="store_true", help='Randomly sample n lines from the monolingual file, instead of taking the first n lines.')

    return ap.parse_args()

def iter_lines(file):
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            yield line.rstrip()

def get_random_indices(file, n, start_offset=1):
    random.seed(42)
    with open(file, 'r', encoding='utf8') as f:
        linecount = sum(1 for _ in f)
    random_idx = set(random.sample(range(start_offset, linecount-1), k=n))

    assert len(random_idx) == n

    return random_idx

def iter_random(file, idces: set):
    with open(file, 'r', encoding='utf8') as f:
        i = 0
        for line in f:
            if i in idces:
                yield line.rstrip()
            i += 1

def remove_duplicate_lines(file):
    tmpname = "temp.txt"
    store = set()
    with open(file, 'r', encoding="utf8") as f, open(tmpname, 'a', encoding="utf8") as t:
        for line in f:
            hashed = hash(line)
            if hashed not in store:
                store.add(hashed)
                t.write(line)
    os.replace(tmpname, file)

def main(args):

    c = 0
    data_set = dict()
    
    # FILESTUFF
    pth = shutil.copytree(Path(args.in_dir) / "raw", Path(args.out_dir) / "raw", dirs_exist_ok=True)

    src_file = Path(pth) / f'train.{args.src_lang}'    
    tgt_file = Path(pth) / f'train.{args.tgt_lang}'

    for src_line, tgt_line in zip(iter_lines(src_file), iter_lines(tgt_file)):
        c += 1
        hashed_mono = hash(src_line + tgt_line)
        if hashed_mono not in data_set:
            data_set[hashed_mono] = [src_line, tgt_line]
    d = 0

    if not args.random_sample:
        for mono_line in iter_lines(args.mono_data):
            hashed_mono = hash(mono_line)
            if hashed_mono not in data_set:
                data_set[hashed_mono] = [mono_line, mono_line]
                d += 1
            # end loop when it exceeds a certain size
            if d >= args.mono_size: break

    else:
        remove_duplicate_lines(args.mono_data)
        random_indices = get_random_indices(args.mono_data, args.mono_size, start_offset=1)

        for mono_line in iter_random(args.mono_data, random_indices):
            hashed_mono = hash(mono_line)
            if hashed_mono not in data_set:
                data_set[hashed_mono] = [mono_line, mono_line]
                d += 1

    data_set = list(data_set.values())
    
    if args.shuffle:
        random.seed(42)
        random.shuffle(data_set)

    c = 0
    with open(src_file, 'w', encoding='utf8') as src_f:
        with open(tgt_file, 'w', encoding='utf8') as tgt_f:
            for src_data, tgt_data in data_set:
                src_f.write(f'{src_data}\n')
                tgt_f.write(f'{tgt_data}\n')
                c += 1
    print(f'Wrote {c} lines to monolingual training split...')
    

if __name__ == '__main__':
    args = set_args()
    print(args)
    # print(vars(args))
    # print(dict(args))
    main(args)
    # random_sample("data/monolingual/TED-en-100.txt", n=10)

    print('Done.')