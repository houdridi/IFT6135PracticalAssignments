import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import RNN, GRU, FullTransformer
from models import make_model as TRANSFORMER

###############################################################################
#
# LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


def load_model(model_info, vocab_size, emb_size=200):
    path = model_info.get_params_path()

    if torch.cuda.is_available():
        print("Loading %s with GPU" % path)
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")

    if model_info.model == 'RNN':
        model = RNN(emb_size=emb_size, hidden_size=model_info.hidden_size,
                    seq_len=model_info.seq_len, batch_size=model_info.batch_size,
                    vocab_size=vocab_size, num_layers=model_info.num_layers,
                    dp_keep_prob=model_info.dp_keep_prob)
    elif model_info.model == 'GRU':
        model = GRU(emb_size=emb_size, hidden_size=model_info.hidden_size,
                    seq_len=model_info.seq_len, batch_size=model_info.batch_size,
                    vocab_size=vocab_size, num_layers=model_info.num_layers,
                    dp_keep_prob=model_info.dp_keep_prob)
    else:
        model = TRANSFORMER(vocab_size=vocab_size, n_units=model_info.hidden_size,
                            n_blocks=model_info.num_layers, dropout=1. - model_info.dp_keep_prob)
        model.batch_size = model_info.batch_size
        model.seq_len = model_info.seq_len
        model.vocab_size = vocab_size

    model.load_state_dict(torch.load(path, map_location=device))
    return model
