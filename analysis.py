#!/usr/bin/env python3
import argparse
import os
import datetime
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import hpbandster.visualization as hpvis
import hpbandster.core.result as hpres
import warnings
import json


def nice_print(left, right):
    print("{:>30} : {}".format(left, right))


def sensitivity_plot(runs, id2conf, cvars, cmap=plt.get_cmap('tab10'), min_budget=0):
    budget_map = {b: i for i, b in enumerate(set([r.budget for r in runs]))}
    handles = [mpatches.Patch(color=cmap(budget_map[b]), label=f'{b}') for b in budget_map.keys()]

    fig = plt.figure(figsize=(8, 8))

    for i, var in enumerate(cvars):
        plt.subplot(len(cvars) // 2 + 1, 2, i + 1)

        for run in runs:
            if run.info is None:
                warnings.warn("Found run with null info! Ignoring it.")
                continue
            if run.budget < min_budget:
                continue
            conf = id2conf[run.config_id]['config']
            acc = np.array([x['target_accuracy'] for x in run.info['single_info']])[:, -2:].mean(axis=1).mean(axis=0)
            plt.scatter(conf[var], acc,
                        color=cmap(budget_map[run.budget]), alpha=0.9)

        plt.title(var)

    plt.legend(handles=handles)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, required=True, help="Final result Pickle file or master directory (for running experiments)")
    parser.add_argument('--mode', type=str, choices=('save', 'show', 'disable'), default='disable', help="Plot mode")
    parser.add_argument('--out-path', type=str, default=None, help="Default output path for plots")
    parser.add_argument('--dpi', type=int, default=150, help="Plot resolution")

    args = parser.parse_args()

    # load run results
    if os.path.isfile(args.result):
        default_out_path = os.path.dirname(os.path.abspath(args.result))
        exp_name = os.path.splitext(os.path.basename(args.result))[0]
        with open(args.result, 'rb') as fp:
            result = pickle.load(fp)
    elif os.path.isdir(args.result):
        default_out_path = args.result
        exp_name = 'exp'
        result = hpres.logged_results_to_HBS_result(args.result)
    else:
        print("No input specified. Use --result")
        return

    save_figs = args.mode is None or args.mode == 'save'
    show_figs = args.mode is None or args.mode == 'show'

    # File path
    out_path = args.out_path or default_out_path

    # Get all executed runs
    all_runs = result.get_all_runs()

    # Get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']

    # Each run contains one or more trainings
    # chosen_accs: list of the chosen best model accuracy (according to the bohb loss) for each single training
    chosen_accs = []

    all_accs = []
    for single_info in inc_run.info['single_info']:
        # All the BOHB losses of this training (one per epoch)
        bohb_losses = np.array(single_info['bohb_losses'])
        # Let's find the best one
        best_index = bohb_losses.argmin()
        # Add the best model (according to the bohb loss) accuracy to chosen_accs
        chosen_accs.append(single_info['target_accuracy'][best_index])
        # Add all the accuracies of all the epochs of this training
        all_accs.append(single_info['target_accuracy'])
    # Get mean accuracy for this run (average the selected models for each training)
    acc = np.array(chosen_accs).mean()
    # Matrix containing ALL the target accuracies of all the epochs of all the trainings of this run
    all_accs = np.array(all_accs)

    # Print best configuration
    print('Best found configuration:')
    for k in inc_config:
        nice_print(k, inc_config[k])
    nice_print('inc_id', '-'.join(map(str, inc_id)))

    print()
    print("Performance:")
    criterion_names = {
        'regression': 'Regression',
        'target_accuracy': 'Trg accuracy',
        'target_entropy_loss': 'Trg entropy',
        'target_div_loss': 'Trg diversity',
        'target_class_loss': 'Trg class. loss',
        'target_silhouette_score': 'Trg Silhouette',
        'target_calinski_harabasz_score': 'Trg Calinski-Harabasz'
    }
    # Print info
    cname = inc_run.info['criterion']
    nice_print(criterion_names.get(cname, cname), f"{inc_loss:.10f}")
    nice_print("Accuracy", f"{acc * 100:.4f} % (mean of each selected model in selected conf run trainings)")
    nice_print("Accuracy", f"{all_accs.max(initial=-1) * 100:.4f} % (best in selected run, you shouldn't know this)")

    print()
    print("Resources:")
    nice_print("Total time", datetime.timedelta(seconds=all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))
    durations = list(map(lambda r: r.time_stamps['finished'] - r.time_stamps['started'], all_runs))
    nice_print("Number of runs", len(all_runs))
    nice_print("Longest run", datetime.timedelta(seconds=max(durations)))
    nice_print("Shortest run", datetime.timedelta(seconds=min(durations)))

    gpu_seconds = sum([r.time_stamps['finished'] - r.time_stamps['started'] for r in all_runs])
    nice_print("GPU time", datetime.timedelta(seconds=gpu_seconds))

    if not (save_figs or show_figs):
        return

    print()
    print("Generating plots")

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)
    if save_figs:
        plt.savefig(os.path.join(out_path, 'loss-over-time_{}.png'.format(exp_name)), dpi=args.dpi)

    # the number of concurrent runs,
    hpvis.concurrent_runs_over_time(all_runs)
    if save_figs:
        plt.savefig(os.path.join(out_path, 'concurrent-runs_{}.png'.format(exp_name)), dpi=args.dpi)

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)
    if save_figs:
        plt.savefig(os.path.join(out_path, 'finished-runs_{}.png'.format(exp_name)), dpi=args.dpi)

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)
    if save_figs:
        plt.savefig(os.path.join(out_path, 'correlation_{}.png'.format(exp_name)), dpi=args.dpi)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
    if save_figs:
        plt.savefig(os.path.join(out_path, 'model-vs-random_{}.png'.format(exp_name)), dpi=args.dpi)

    sensitivity_plot(all_runs, id2conf, cvars=('disc.num_fc_layers', 'disc.hidden_size_log', 'disc.dropout',
                                               'net.bottleneck_size_log', 'base.weight_da'))
    if save_figs:
        plt.savefig(os.path.join(out_path, 'sensitivity_{}.png'.format(exp_name)), dpi=args.dpi)

    if show_figs:
        plt.show()


if __name__ == '__main__':
    main()
