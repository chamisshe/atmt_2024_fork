import argparse
import collections
import logging
import os
import sys
import re
import pickle

# establish link to seq2seq dir
# scripts_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = os.path.join(scripts_dir, "..")
# sys.path.append(base_dir)

from seq2seq import utils
from seq2seq.data.dictionary import Dictionary

SPACE_NORMALIZER = re.compile("\s+")


def word_tokenize(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def get_args():
    parser = argparse.ArgumentParser('Data pre-processing)')
    parser.add_argument('--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('--target-lang', default=None, metavar='TGT', help='target language')

    parser.add_argument('--split', default='data-bin', metavar='DIR', help='destination dir')
    parser.add_argument('--dest-dir', default='data-bin', metavar='DIR', help='destination dir')

    parser.add_argument('--vocab-bpe', default=None, type=str, help='path to dictionary')
    parser.add_argument('--quiet', action='store_true', help='no logging')

    return parser.parse_args()


def main(args):
    src_dict = Dictionary.load(args.vocab_bpe)
    if not args.quiet:
        logging.info('Loaded a source dictionary ({}) with {} words'.format(args.target_lang, len(src_dict)))


    tgt_dict = Dictionary.load(args.vocab_bpe)
    if not args.quiet:
        logging.info('Loaded a target dictionary ({}) with {} words'.format(args.target_lang, len(tgt_dict)))

    def make_split_datasets(lang, dictionary):
        make_binary_dataset(args.split + '.' + lang, os.path.join(args.dest_dir, 'train.' + lang),
                                dictionary)
        
    make_split_datasets(args.source_lang, src_dict)
    make_split_datasets(args.target_lang, tgt_dict)


def build_dictionary(filenames, tokenize=word_tokenize):
    dictionary = Dictionary()
    for filename in filenames:
        with open(filename, 'r') as file:
            for line in file:
                for symbol in word_tokenize(line.strip()):
                    dictionary.add_word(symbol)
                dictionary.add_word(dictionary.eos_word)
    return dictionary


def make_binary_dataset(input_file, output_file, dictionary, tokenize=word_tokenize, append_eos=True):
    nsent, ntok = 0, 0
    unk_counter = collections.Counter()

    def unk_consumer(word, idx):
        if idx == dictionary.unk_idx and word != dictionary.unk_word:
            unk_counter.update([word])

    tokens_list = []
    with open(input_file, 'r') as inf:
        for line in inf:
            tokens = dictionary.binarize(line.strip(), word_tokenize, append_eos, consumer=unk_consumer)
            nsent, ntok = nsent + 1, ntok + len(tokens)
            tokens_list.append(tokens.numpy())

    with open(output_file, 'wb') as outf:
        pickle.dump(tokens_list, outf, protocol=pickle.DEFAULT_PROTOCOL)
        if not args.quiet:
            logging.info('Built a binary dataset for {}: {} sentences, {} tokens, {:.3f}% replaced by unknown token'.format(
            input_file, nsent, ntok, 100.0 * sum(unk_counter.values()) / ntok, dictionary.unk_word))


if __name__ == '__main__':
    args = get_args()
    if not args.quiet:
        utils.init_logging(args)
        logging.info('COMMAND: %s' % ' '.join(sys.argv))
        logging.info('Arguments: {}'.format(vars(args)))
    main(args)
