import torch
import numpy as np
from ptb_helpers import ptb_raw_data, load_model
from itertools import product
import os

torch.manual_seed(25)


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

    def get_params_path(self, folder_prefix='_save_best_0'):
        params_file = 'best_params.pt'
        folder_path = '%s_%s_model=%s_optimizer=%s_initial_lr=%s_batch_size=%s_' \
                      'seq_len=%s_hidden_size=%s_num_layers=%s_dp_keep_prob=%s%s' \
                      % (self.model, self.optimizer, self.model, self.optimizer, self.initial_lr,
                         self.batch_size, self.seq_len, self.hidden_size,
                         self.num_layers, self.dp_keep_prob, folder_prefix)
        return os.path.join(self.section, folder_path, params_file)


def generate_sentences(model_info, seq_len, batch_size=10, start_word="<eos>", results_folder='./5_results'):

    train_data, valid_data, test_data, word_to_id, id_2_word = ptb_raw_data(data_path='data')
    vocab_size = len(word_to_id)
    start_input = torch.LongTensor(np.ones(batch_size) * word_to_id[start_word])
    model = load_model(model_info, vocab_size)

    hidden = torch.zeros(model.num_layers, batch_size, model.hidden_size)
    samples = model.generate(start_input, hidden, seq_len).numpy()
    generated_sentences = []
    for i in range(batch_size):
        sentence = " ".join([starting_word] + [id_2_word[s] for s in samples[:, i]])
        generated_sentences.append(sentence)

    with open(os.path.join(results_folder, '%s_%s_generated_samples.txt' % (model_info.model, seq_len)), 'w') as f:
        for sentence in generated_sentences:
            f.write("%s\n" % sentence)


if __name__ == "__main__":
    seq_lens = [35, 70]
    generations_per_seq_len = 10
    starting_word = "<eos>"
    models = [ModelInfo('RNN', 'ADAM', 0.0001, 20, 35, 1500, 2, 0.35),
              ModelInfo('GRU', 'SGD_LR_SCHEDULE', 10, 20, 35, 1500, 2, 0.35)]

    print("Generating sentences")
    for model, seq_len in product(models, seq_lens):
        generate_sentences(model, seq_len, generations_per_seq_len, starting_word)
