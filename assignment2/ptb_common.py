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


def init_device():
    if torch.cuda.is_available():
        print("Using the GPU")
        device = torch.device("cuda")
    else:
        print("WARNING: You are about to run on cpu, and this will likely run out \
          of memory. \n You can try setting batch_size=1 to reduce memory usage")
        device = torch.device("cpu")
    return device


def load_model(model_info, device, vocab_size, emb_size=200, load_on_device=True):
    params_path = model_info.get_params_path()

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

    if load_on_device:
        model = model.to(device)
    model.load_state_dict(torch.load(params_path, map_location=device))
    return model


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.

    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.

    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


class Batch:
    """Data processing for the transformer. This class adds a mask to the data."""

    def __init__(self, x, pad=-1):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        """Create a mask to hide future words."""

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


class ModelInfo:
    def __init__(self, model, optimizer, initial_lr, batch_size, seq_len, hidden_size, num_layers, dp_keep_prob, section='4_1'):
        self.model = model
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob
        self.section = section

    def get_folder_path(self, folder_prefix='_save_best_0'):
        results_folder = '%s_%s_model=%s_optimizer=%s_initial_lr=%s_batch_size=%s_' \
                      'seq_len=%s_hidden_size=%s_num_layers=%s_dp_keep_prob=%s%s' \
                      % (self.model, self.optimizer, self.model, self.optimizer, self.initial_lr,
                         self.batch_size, self.seq_len, self.hidden_size,
                         self.num_layers, self.dp_keep_prob, folder_prefix)
        return os.path.join(self.section, results_folder)

    def get_params_path(self, folder_prefix='_save_best_0'):
        params_file = 'best_params.pt'
        return os.path.join(self.get_folder_path(folder_prefix), params_file)


def normalize_times(model_info: ModelInfo, scale):
    results_path = model_info.get_folder_path()
    learning_curves_raw = np.load(os.path.join(results_path, 'learning_curves.npy'))
    learning_curves = learning_curves_raw[()]
    learning_curves['times'] = [t*scale for t in learning_curves['times']]

    with open(os.path.join(results_path, 'log.txt'), 'w') as f_:
        for epoch in range(40):
            log_str = 'epoch: ' + str(epoch) + '\t' \
                    + 'train ppl: ' + str(learning_curves['train_ppls'][epoch]) + '\t' \
                    + 'val ppl: ' + str(learning_curves['val_ppls'][epoch])  + '\t' \
                    + 'best val: ' + str(learning_curves['best_vals'][epoch]) + '\t' \
                    + 'time (s) spent in epoch: ' + str(learning_curves['times'][epoch])
            f_.write(log_str+ '\n')
    np.save(os.path.join(results_path, 'learning_curves.npy'), learning_curves_raw)


# normalize_times(ModelInfo(
#     model='RNN',
#     optimizer='ADAM',
#     initial_lr=0.0001,
#     batch_size=20,
#     seq_len=35,
#     hidden_size=1500,
#     num_layers=2,
#     dp_keep_prob=0.35
# ), scale=1)

