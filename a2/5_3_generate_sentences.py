import torch
import numpy as np
from ptb_common import ptb_raw_data, load_model, init_device, ModelInfo
from itertools import product
import os

# See for reproducibility
torch.manual_seed(25)


def generate_sentences(model_info, device, seq_len, batch_size=10, start_word="<eos>", results_folder='./5_results'):

    train_data, valid_data, test_data, word_to_id, id_2_word = ptb_raw_data(data_path='data')
    vocab_size = len(word_to_id)
    start_input = torch.LongTensor(np.ones(batch_size) * word_to_id[start_word])
    model = load_model(model_info, device, vocab_size=vocab_size, load_on_device=False)

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
    # Since data is fed into models in a continuous fahsion, can use eos to signal the beginning of a new sentence
    starting_word = "<eos>"
    # Models from section 4.1
    models = [ModelInfo('RNN', 'ADAM', 0.0001, 20, 35, 1500, 2, 0.35),
              ModelInfo('GRU', 'SGD_LR_SCHEDULE', 10, 20, 35, 1500, 2, 0.35)]

    print("Generating sentences")
    device = init_device()
    for m, s in product(models, seq_lens):
        generate_sentences(m, device, s, generations_per_seq_len, starting_word)
