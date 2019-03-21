import numpy as np
import os
import re
import itertools
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_FOLDER = './results'


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
        max_valid_ppls_idx = np.argmax(self.val_ppls)
        self.best_valid_ppl = self.val_ppls[max_valid_ppls_idx]
        self.best_valid_train_ppl = self.train_ppls[max_valid_ppls_idx]

    def __str__(self) -> str:
        return "%s opt=%s init_lr=%s bat_sx=%s seq_len=%s h_sx=%s n_lyrs=%s dp_kp_prb=%s" % \
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
    return [ExperimentInfo(section, result_folder) for result_folder in os.listdir(section)]


def plot_experiment_performance(experiment: ExperimentInfo):

    # Validation + Train PPL vs Epochs
    fig = plt.figure()
    plt.plot(experiment.val_ppls, 'b-o', label="Val PPL")
    plt.plot(experiment.train_ppls, '--o', label="Train PPL")
    plt.legend()
    fig.suptitle("%s\nPPL vs. Epochs" % experiment, fontsize=10)
    plt.ylabel("PPL")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(experiment.result_folder, 'ppl_vs_epochs.png'), bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(RESULTS_FOLDER, 'individual_experiments', '%s_ppl_vs_epochs.png' % experiment.full_name),
                bbox_inches='tight', pad_inches=0.2)
    plt.clf()
    plt.close()

    # Validation + Train PPL vs Wall-Clock Time
    fig = plt.figure()
    plt.plot(experiment.wct, experiment.val_ppls, 'b-o', label="Val PPL")
    plt.plot(experiment.wct, experiment.train_ppls, '--o', label="Train PPL")
    plt.legend()
    fig.suptitle("%s\nPPL vs. Wall-Clock Time" % experiment, fontsize=10)
    plt.ylabel("PPL")
    plt.xlabel("Wall-Clock Time")
    plt.savefig(os.path.join(experiment.result_folder, 'perf_vs_wct.png'), bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(RESULTS_FOLDER, 'individual_experiments', '%s_perf_vs_wct.png' % experiment.full_name),
                bbox_inches='tight', pad_inches=0.2)
    plt.clf()
    plt.close()


def plot_section_results(section: str, section_experiments: [ExperimentInfo], legend_size=8):
    colors = ['C' + str(i + 1) for i in range(len(section_experiments))]

    if not section_experiments:
        return

    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(section_experiments):
        # Validation + Train PPL vs Epochs
        plt.plot(experiment.val_ppls, '-o', color=colors[i], label="%s Val" % experiment)
        plt.plot(experiment.train_ppls, '--o', color=colors[i], alpha=0.6, label="%s Train" % experiment)

    plt.legend(prop={'size': legend_size})
    plt.title("%s PPL vs. Epochs" % section)
    plt.ylabel("PPL")
    plt.xlabel("Epochs")
    plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_epochs.png' % section), bbox_inches='tight',
                pad_inches=0.2)
    plt.clf()
    plt.close()

    plt.figure(figsize=(10, 6))
    for i, experiment in enumerate(section_experiments):
        # Validation + Train PPL vs Wall-Clock Time
        plt.plot(experiment.wct, experiment.val_ppls, '-o', color=colors[i], label="%s Val" % experiment)
        plt.plot(experiment.wct, experiment.train_ppls, '--o', color=colors[i], alpha=0.6, label="%s Train" % experiment)

    plt.legend(prop={'size': legend_size})
    plt.title("%s PPL vs. Wall-Clock Time" % section)
    plt.ylabel("PPL")
    plt.xlabel("Wall-Clock Time")
    plt.savefig(os.path.join(RESULTS_FOLDER, '%s_perf_vs_wct.png' % section), bbox_inches='tight',
                pad_inches=0.2)
    plt.clf()
    plt.close()


def plot_valid_perf_by_optimizer(all_experiments: [ExperimentInfo], legend_size=8):
    optimizers = ['SGD_LR_SCHEDULE', 'SGD', 'ADAM']
    for optimizer in optimizers:
        opt_experiments = [e for e in all_experiments if e.optimizer == optimizer]

        if not opt_experiments:
            continue

        plt.figure(figsize=(10, 6))
        colors = ['C' + str(i + 1) for i in range(len(opt_experiments))]
        for i, experiment in enumerate(opt_experiments):
            plt.plot(experiment.val_ppls, '-o', color=colors[i], label=experiment.str_except_optimizer())

        plt.legend(prop={'size': legend_size})
        plt.title("%s PPL vs. Epochs" % optimizer)
        plt.ylabel("PPL")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_epochs.png' % optimizer), bbox_inches='tight',
                    pad_inches=0.2)
        plt.clf()
        plt.close()

        plt.figure(figsize=(10, 6))
        for i, experiment in enumerate(opt_experiments):
            plt.plot(experiment.wct, experiment.val_ppls, '-o', color=colors[i], label=experiment.str_except_optimizer())

        plt.legend(prop={'size': legend_size})
        plt.title("%s PPL vs. Wall-Clock Time" % optimizer)
        plt.ylabel("PPL")
        plt.xlabel("Wall-Clock Time")
        plt.savefig(os.path.join(RESULTS_FOLDER, '%s_perf_vs_wct.png' % optimizer), bbox_inches='tight',
                    pad_inches=0.2)
        plt.clf()
        plt.close()


def plot_valid_perf_by_architecture(all_experiments: [ExperimentInfo], legend_size=8):
    models = ['RNN', 'GRU', 'TRANSFORMER']
    for model in models:
        model_experiments = [e for e in all_experiments if e.model == model]

        if not model_experiments:
            continue

        colors = ['C' + str(i + 1) for i in range(len(model_experiments))]

        plt.figure(figsize=(10, 6))
        for i, experiment in enumerate(model_experiments):
            plt.plot(experiment.val_ppls, '-o', color=colors[i], label=experiment.str_except_model())
        plt.legend(prop={'size': legend_size})
        plt.title("%s PPL vs. Epochs" % model)
        plt.ylabel("PPL")
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(RESULTS_FOLDER, '%s_ppl_vs_epochs.png' % model), bbox_inches='tight',
                    pad_inches=0.2)
        plt.clf()
        plt.close()

        plt.figure(figsize=(10, 6))
        for i, experiment in enumerate(model_experiments):
            plt.plot(experiment.wct, experiment.val_ppls, '-o', color=colors[i], label=experiment.str_except_model())

        plt.legend(prop={'size': legend_size})
        plt.title("%s PPL vs. Wall-Clock Time" % model)
        plt.ylabel("PPL")
        plt.xlabel("Wall-Clock Time")
        plt.savefig(os.path.join(RESULTS_FOLDER, '%s_perf_vs_wct.png' % model), bbox_inches='tight',
                    pad_inches=0.2)
        plt.clf()
        plt.close()


def graph_all_results():
    sections = ['4_1', '4_2', '4_3']
    experiments = list(itertools.chain(*[get_experiments(q) for q in sections]))

    # 1) Plot performance for each experiment
    for exp in experiments:
        plot_experiment_performance(exp)

    # 2) Make table summarizing results
    df = pd.DataFrame(columns=['model', 'optimizer', 'initial_lr', 'batch_size', 'seq_len', 'hidden_size', 'num_layers',
                               'dp_keep_prob'])
    for exp in experiments:
        df = df.append({
            'section': exp.section,
            'model': exp.model,
            'optimizer': exp.optimizer,
            'initial_lr': exp.initial_lr,
            'batch_size': exp.batch_size,
            'seq_len': exp.seq_len,
            'hidden_size': exp.hidden_size,
            'num_layers': exp.num_layers,
            'dp_keep_prob': exp.dp_keep_prob,
            'train_val': exp.best_valid_train_ppl,
            'val_ppl': exp.best_valid_ppl,
            'avg_wct': np.mean(exp.times).round(2)
        }, ignore_index=True)

    df = df.sort_values(by=['model', 'optimizer', 'val_ppl'])
    for section in sections:
        df.loc[df['section'] == section].drop(['section'], axis=1) \
            .to_csv(os.path.join(RESULTS_FOLDER, '{}_results.csv'.format(section)), index=False)

    df.drop(['section'], axis=1).to_csv(os.path.join(RESULTS_FOLDER, 'results.csv'), index=False)

    # 3) Plot section results
    for section in sections:
        section_experiments = [e for e in experiments if e.section == section]
        plot_section_results(section, section_experiments)

    # 4) Validation ppl curves by optimizer
    plot_valid_perf_by_optimizer(experiments)

    # 5) Validation ppl curves by architecture
    plot_valid_perf_by_architecture(experiments)


if __name__ == '__main__':
    graph_all_results()









