import numpy as np
import os
import re
import itertools
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FOLDER = './4_plots_and_tables'
# y_lim for rnn + sgd or composite graphs because rnn in 4.2 also reaches low 2000 at end of training
rnn_sgd_y_lim = 3000


class ExperimentInfo:

    def __init__(self, section, folder_path):
        self.section = section
        self.full_name = folder_path
        self.result_folder = os.path.join(section, folder_path)
        self.model = re.search('model=(.*)_optimizer', folder_path, re.IGNORECASE).group(1)
        self.optimizer = re.search('optimizer=(.*)_initial_lr', folder_path, re.IGNORECASE).group(1)
        self.initial_lr = re.search('initial_lr=(.*)_batch_size', folder_path, re.IGNORECASE).group(1)
        self.batch_size = re.search('batch_size=(.*)_seq_len', folder_path, re.IGNORECASE).group(1)
        self.seq_len = re.search('seq_len=(.*)_hidden_size', folder_path, re.IGNORECASE).group(1)
        self.hidden_size = re.search('hidden_size=(.*)_num_layers', folder_path, re.IGNORECASE).group(1)
        self.num_layers = re.search('num_layers=(.*)_dp_keep_prob', folder_path, re.IGNORECASE).group(1)
        self.dp_keep_prob = re.search('dp_keep_prob=(.*)_save_best', folder_path, re.IGNORECASE).group(1)
        self.results = np.load(os.path.join(self.result_folder, 'learning_curves.npy'))[()]

        self.val_ppls = self.results['val_ppls']
        self.train_ppls = self.results['train_ppls']
        self.times = self.results['times']
        self.wct = [np.sum(self.times[0:t+1]) for t in range(len(self.times))]
        min_valid_ppls_idx = np.argmin(self.val_ppls)
        self.best_valid_ppl = self.val_ppls[min_valid_ppls_idx]
        self.best_valid_train_ppl = self.train_ppls[min_valid_ppls_idx]

    def __str__(self) -> str:
        return "%s opt=%s init_lr=%s bat_sx=%s seq_len=%s h_sx=%s n_lyrs=%s dp_kp_prb=%s" % \
               (self.model, self.optimizer, self.initial_lr, self.batch_size,
                self.seq_len, self.hidden_size, self.num_layers, self.dp_keep_prob)

    def split_name(self) -> str:
        return "%s opt=%s init_lr=%s bat_sx=%s\nseq_len=%s h_sx=%s n_lyrs=%s dp_kp_prb=%s" % \
               (self.model, self.optimizer, self.initial_lr, self.batch_size,
                self.seq_len, self.hidden_size, self.num_layers, self.dp_keep_prob)

    def str_except_optimizer(self) -> str:
        return "%s init_lr=%s bat_sx=%s seq_len=%s h_sx=%s n_lyrs=%s dp_kp_prb=%s" % \
               (self.model, self.initial_lr, self.batch_size,
                self.seq_len, self.hidden_size, self.num_layers, self.dp_keep_prob)

    def str_except_model(self) -> str:
        return "%s init_lr=%s bat_sx=%s seq_len=%s h_sx=%s n_lyrs=%s dp_kp_prb=%s" % \
               (self.optimizer, self.initial_lr, self.batch_size,
                self.seq_len, self.hidden_size, self.num_layers, self.dp_keep_prob)


def get_experiments(section):
    return [ExperimentInfo(section, result_folder) for result_folder in os.listdir(section)
            if result_folder != 'extra']


def plot_experiment_performance(experiment: ExperimentInfo, y_lim=None):

    # Validation + Train PPL vs Epochs
    fig = plt.figure()
    plt.plot(experiment.val_ppls, 'b-o', label="Val PPL")
    plt.plot(experiment.train_ppls, '--o', mfc='none', label="Train PPL")
    plt.legend()
    fig.suptitle("%s\nPPL vs. Epochs" % experiment.split_name(), fontsize=10)
    plt.ylabel("PPL")
    plt.xlabel("Epochs")
    if y_lim:
        plt.ylim(0, y_lim)
    plt.savefig(os.path.join(experiment.result_folder, 'ppl_vs_epochs.png'), bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(RESULTS_FOLDER, 'experiments', '%s_ppl_vs_epochs.png' % experiment.full_name),
                bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()

    # Validation + Train PPL vs Wall-Clock Time
    fig = plt.figure()
    plt.plot(experiment.wct, experiment.val_ppls, 'b-o', label="Val PPL")
    plt.plot(experiment.wct, experiment.train_ppls, '--o', mfc='none', label="Train PPL")
    plt.legend()
    fig.suptitle("%s\nPPL vs. Wall-Clock Time" % experiment.split_name(), fontsize=10)
    plt.ylabel("PPL")
    plt.xlabel("Wall-Clock Time")
    if y_lim:
        plt.ylim(0, y_lim)
    plt.savefig(os.path.join(experiment.result_folder, 'ppl_vs_wct.png'), bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(RESULTS_FOLDER, 'experiments', '%s_ppl_vs_wct.png' % experiment.full_name),
                bbox_inches='tight', pad_inches=0.1)
    plt.clf()
    plt.close()


def is_above_lim(experiments, y_lim):
    for e in experiments:
        if any(ppl > y_lim for ppl in e.val_ppls) or any(ppl > y_lim for ppl in e.train_ppls):
            return True
    return False


def plot_section_results(section: str, section_experiments: [ExperimentInfo], legend_size=8, y_lim=1000):
    colors = ['C' + str((i + 1) % 10) for i in range(len(section_experiments))]

    if not section_experiments:
        return

    clip_y = is_above_lim(section_experiments, y_lim)

    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(section_experiments):
        # Validation + Train PPL vs Epochs
        plt.plot(experiment.val_ppls, '-o', color=colors[i], label="%s Val" % experiment)
        plt.plot(experiment.train_ppls, '--o', color=colors[i], alpha=0.6, mfc='none', label="%s Train" % experiment)

    plt.legend(prop={'size': legend_size})
    plt.title("%s PPL vs. Epochs" % section)
    plt.ylabel("PPL")
    plt.xlabel("Epochs")
    if clip_y:
        plt.ylim(0, y_lim)
    plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_epochs.png' % section), bbox_inches='tight',
                pad_inches=0.2)
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(section_experiments):
        # Validation + Train PPL vs Wall-Clock Time
        plt.plot(experiment.wct, experiment.val_ppls, '-o', color=colors[i], label="%s Val" % experiment)
        plt.plot(experiment.wct, experiment.train_ppls, '--o', color=colors[i], mfc='none', alpha=0.6, label="%s Train" % experiment)

    plt.legend(prop={'size': legend_size})
    plt.title("%s PPL vs. Wall-Clock Time" % section)
    plt.ylabel("PPL")
    plt.xlabel("Wall-Clock Time")
    if clip_y:
        plt.ylim(0, y_lim)
    plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_wct.png' % section), bbox_inches='tight',
                pad_inches=0.2)
    plt.clf()
    plt.close()


def plot_valid_perf_by_optimizer(all_experiments: [ExperimentInfo], legend_size=8):
    optimizers = ['SGD_LR_SCHEDULE', 'SGD', 'ADAM']
    for optimizer in optimizers:
        opt_experiments = [e for e in all_experiments if e.optimizer == optimizer]

        opt_experiments.sort(key=lambda x: x.model)
        y_lim = rnn_sgd_y_lim if optimizer == 'SGD' else (600 if optimizer == 'ADAM' else 1000)

        if not opt_experiments:
            continue

        plt.figure(figsize=(10, 6))
        colors = ['C' + str((i + 1) % 10) for i in range(len(opt_experiments))]
        for i, experiment in enumerate(opt_experiments):
            plt.plot(experiment.val_ppls, '-o', color=colors[i], alpha=0.7, label=experiment.str_except_optimizer())

        plt.legend(prop={'size': legend_size})
        plt.title("%s Validation PPL vs. Epochs" % optimizer)
        plt.ylabel("PPL")
        plt.xlabel("Epochs")
        plt.ylim(0, y_lim)
        plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_epochs.png' % optimizer), bbox_inches='tight',
                    pad_inches=0.2)
        plt.clf()
        plt.close()

        plt.figure(figsize=(10, 6))
        for i, experiment in enumerate(opt_experiments):
            plt.plot(experiment.wct, experiment.val_ppls, '-o', color=colors[i], alpha=0.7, label=experiment.str_except_optimizer())

        plt.legend(prop={'size': legend_size})
        plt.title("%s Validation PPL vs. Wall-Clock Time" % optimizer)
        plt.ylabel("PPL")
        plt.xlabel("Wall-Clock Time")
        plt.ylim(0, y_lim)
        plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_wct.png' % optimizer), bbox_inches='tight',
                    pad_inches=0.2)
        plt.clf()
        plt.close()


def plot_valid_perf_for_architecture(model_name, model_experiments, y_lim=None, legend_size=8):
    model_experiments.sort(key=lambda x: x.optimizer)
    colors = ['C' + str((i + 1) % 10) for i in range(len(model_experiments))]

    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(model_experiments):
        plt.plot(experiment.val_ppls, '-o', color=colors[i], alpha=0.7, label=experiment.str_except_model())
    plt.legend(prop={'size': legend_size})
    plt.title("%s Validation PPL vs. Epochs" % model_name)
    plt.ylabel("PPL")
    plt.xlabel("Epochs")
    if y_lim:
        plt.ylim(0, y_lim)
    plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_epochs.png' % model_name.replace(' ', '_')), bbox_inches='tight',
                pad_inches=0.2)
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(model_experiments):
        plt.plot(experiment.wct, experiment.val_ppls, '-o', color=colors[i], alpha=0.7,
                 label=experiment.str_except_model())

    plt.legend(prop={'size': legend_size})
    plt.title("%s Validation PPL vs. Wall-Clock Time" % model_name)
    plt.ylabel("PPL")
    plt.xlabel("Wall-Clock Time")
    if y_lim:
        plt.ylim(0, y_lim)
    plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_wct.png' % model_name.replace(' ', '_')), bbox_inches='tight',
                pad_inches=0.2)
    plt.clf()
    plt.close()


def plot_valid_perf_by_architecture(all_experiments: [ExperimentInfo]):
    models = ['RNN', 'GRU', 'TRANSFORMER']
    for model in models:
        model_experiments = [e for e in all_experiments if e.model == model]
        y_lim = rnn_sgd_y_lim if model == 'RNN' else (1000 if model == 'TRANSFORMER' else None)
        if not model_experiments:
            continue
        plot_valid_perf_for_architecture(model, model_experiments, y_lim)

    # Extra plot for rnn architectures excluding outlier
    rnn_experiments_without_outlier = [e for e in all_experiments if e.model == 'RNN' if not (e.optimizer == 'SGD'
                                       and e.initial_lr == '0.0001')]
    plot_valid_perf_for_architecture('RNN without outlier', rnn_experiments_without_outlier)


def graph_all_results():
    sections = ['4_1', '4_2', '4_3']
    experiments = list(itertools.chain(*[get_experiments(q) for q in sections]))

    # 1) Plot performance for each experiment
    for exp in experiments:
        print('Graphing performance %s' % exp)
        plot_experiment_performance(exp, y_lim=1500 if exp.model == 'TRANSFORMER' else None)

    # 2) Make table summarizing results
    print("Creating summary tables")
    df = pd.DataFrame(columns=['model', 'optimizer', 'initial_lr', 'batch_size', 'seq_len', 'hidden_size', 'num_layers',
                               'dp_keep_prob', 'avg_wct', 'train_val', 'val_ppl', 'section'])
    for exp in experiments:
        df = df.append({
            'model': exp.model,
            'optimizer': exp.optimizer,
            'initial_lr': exp.initial_lr,
            'batch_size': exp.batch_size,
            'seq_len': exp.seq_len,
            'hidden_size': exp.hidden_size,
            'num_layers': exp.num_layers,
            'dp_keep_prob': exp.dp_keep_prob,
            'avg_wct': np.mean(exp.times).round(2),
            'train_val': exp.best_valid_train_ppl,
            'val_ppl': exp.best_valid_ppl,
            'section': exp.section,
        }, ignore_index=True)

    df = df.sort_values(by=['model', 'optimizer', 'val_ppl'])
    df = df.round(2)
    for section in sections:
        df.loc[df['section'] == section].drop(['section'], axis=1) \
            .to_csv(os.path.join(RESULTS_FOLDER, '{}_results_table.csv'.format(section)), index=False)

    df.to_csv(os.path.join(RESULTS_FOLDER, 'results_table.csv'), index=False)

    # 3) Plot section results
    print("Graphing results per section")
    for section in sections[:-1]:
        section_experiments = [e for e in experiments if e.section == section]
        plot_section_results(section, section_experiments, y_lim=rnn_sgd_y_lim if section == '4_2' else 1000)

    # 4) Validation ppl curves by optimizer
    print("Graphing results per optimizer")
    plot_valid_perf_by_optimizer(experiments)

    # 5) Validation ppl curves by architecture
    print("Graphing results per architecture")
    plot_valid_perf_by_architecture(experiments)


if __name__ == '__main__':
    graph_all_results()









