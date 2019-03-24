import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from ptb_common import load_model, init_device, ptb_raw_data, ModelInfo, Batch, ptb_iterator, repackage_hidden
from models import RNN, GRU
from models import make_model as TRANSFORMER
from sklearn.preprocessing import minmax_scale


def compute_grad_per_timestep(model, device, data, loss_fn):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    model.eval()

    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):

        hidden = model.init_hidden()
        hidden = hidden.to(device)

        inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        model.zero_grad()
        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)
        # LOSS COMPUTATION
        last_loss = loss_fn(outputs[-1], targets[-1])

        # Compute gradient with respect to last loss. Average gradients in batch
        grads = torch.empty(0).to(device)
        for layer_states in model.state_history:
            batch_grads = torch.autograd.grad(last_loss, layer_states, retain_graph=True)
            avg_grads = [torch.mean(bg, 0) for bg in batch_grads]
            # Gradients for each layer concatenated into a single vector
            grads = torch.cat((grads, torch.stack(avg_grads)), dim=1)

        return [s.norm() for s in grads]


def plot_loss_per_step(model_infos, grads_by_model):

    results_folder = './5_results'
    x = np.arange(35) + 1

    for model_info, grads in zip(model_infos, grads_by_model):
        grads = minmax_scale(grads)
        plt.plot(x, grads, '-o', label=model_info.model)
    plt.title("$\\nabla h_t L_T$ Gradient Normal Vs. Timestep")
    plt.ylabel("Gradient Norm (Scaled)")
    plt.xlabel("Timestep")
    plt.legend()
    file_name = 'grads_per_timestep.png'
    plt.savefig(os.path.join(results_folder, file_name), bbox_inches='tight', pad_inches=0.2)
    plt.clf()
    plt.close()


def compute_grad_per_timestep_by_model():
    device = init_device()
    train_data, valid_data, test_data, word_to_id, id_2_word = ptb_raw_data(data_path='data')
    vocab_size = len(word_to_id)
    loss_fn = nn.CrossEntropyLoss()
    # # Models from 4_1
    model_infos = [ModelInfo('RNN', 'ADAM', 0.0001, 20, 35, 1500, 2, 0.35),
                   ModelInfo('GRU', 'SGD_LR_SCHEDULE', 10, 20, 35, 1500, 2, 0.35)]

    grads_by_model = []
    for model_info in model_infos:
        model = load_model(model_info, device, vocab_size)
        model.track_state_history = True
        grads_per_step = compute_grad_per_timestep(model, device, valid_data, loss_fn)
        grads_by_model.append(grads_per_step)

    # np.save('grads_by_model.npy', grads_by_model)
    # grads_by_model = np.load('grads_by_model.npy')
    plot_loss_per_step(model_infos, grads_by_model)


if __name__ == '__main__':
    print('5.2 - Gradient per timestep')
    compute_grad_per_timestep_by_model()



