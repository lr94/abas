#!/usr/bin/env python3
import argparse
import logging

import pickle
import numpy as np
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers.bohb import BOHB
from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as csh
from hpbandster.core.worker import Worker
import hpbandster.core.result as hpres

from logger import Logger
from utils import *
from train_model import run_training
from model_selection import ModelCriterion


class AbasWorker(Worker):

    def __init__(self, run_id: str, gpu: int, source, target, net, load_workers, max_iter, logdir, ds_root,
                 run_n_avg, no_tqdm, da_method, model_criterion: ModelCriterion, run_model_criterion: ModelCriterion,
                 kill_diverging, *args, **kwargs):
        super().__init__(run_id, args, **kwargs)
        self.run_id = run_id
        self.gpu = gpu
        self.source = source
        self.target = target
        self.net = net
        self.load_workers = load_workers
        self.max_iter = max_iter
        self.logdir = logdir
        self.ds_root = ds_root
        self.run_n_avg = run_n_avg
        self.da_method = da_method
        self.model_criterion = model_criterion
        self.run_model_criterion = run_model_criterion
        self.kill_diverging = kill_diverging

        self.no_tqdm = no_tqdm

    @staticmethod
    def get_configspace(hp: dict):
        config_space = ConfigurationSpace()

        # Add fixed hyperparameters (they are not going to be optimised by ABAS)
        for hp_name in hp:
            config_space.add_hyperparameter(csh.Constant(hp_name, hp[hp_name]))

        # Discriminator HPs (to be optimised)
        config_space.add_hyperparameter(csh.UniformIntegerHyperparameter('disc.num_fc_layers', lower=2, upper=7))
        config_space.add_hyperparameter(csh.UniformIntegerHyperparameter('disc.hidden_size_log', lower=6, upper=12))
        config_space.add_hyperparameter(csh.UniformFloatHyperparameter('disc.dropout', lower=0., upper=1.))

        # Other HPSs (to be optimised)
        config_space.add_hyperparameter(csh.UniformIntegerHyperparameter('net.bottleneck_size_log', lower=6, upper=10))
        config_space.add_hyperparameter(csh.UniformFloatHyperparameter('base.weight_da', lower=0., upper=2.))

        return config_space

    def compute(self, config_id, config, budget, working_directory):
        # This "run name" refers to an actual single training.
        run_name_base = 'cfg{}_bdg{:.3f}_{}'.format('-'.join(map(lambda x: str(x), config_id)), budget,
                                                    dict_shortened_summary(config))

        single_training_losses = []
        single_training_infos = []
        # We *might* want to run multiple trainings for each configuration and average them.
        # we are not doing so in the paper, run_n_avg is left to 1
        for run_i in range(self.run_n_avg):
            run_name = f"{run_name_base}_run{run_i}" if self.run_n_avg > 1 else run_name_base
            print(f"Starting run {run_name}")

            # Run the actual model training
            metrics = run_training(
                source=self.source,
                target=self.target,
                dataset_root=self.ds_root,
                net_name=self.net,
                da_method=self.da_method,
                config=config,
                max_iter=int(self.max_iter),
                stop_iter=int(budget),
                test_iter=100,
                run_name=run_name,
                gpu_id=self.gpu,
                load_workers=self.load_workers,
                test_src=True,
                use_tqdm=not self.no_tqdm,
                logdir=os.path.join(self.logdir, self.run_id),
                kill_diverging=self.kill_diverging
            )

            # Compute BOHB losses according to the selected criterion. These are to compare across different runs
            bohb_losses = self.model_criterion(metrics)
            # This criterion is to compare snapshots within a single run, and it could be different from the other
            # criterion (in the paper it is)
            run_modsel_losses = self.run_model_criterion(metrics)

            # Print info
            print("Run Model Selection losses: ", run_modsel_losses)
            # Select the best model of this training according to run model selection
            best_index = run_modsel_losses.argmin()
            best_loss = run_modsel_losses[best_index]
            best_bohb_loss = bohb_losses[best_index]
            best_iter = metrics['iter'][best_index]
            print(f"Best run modsel loss: {best_loss} at iteration {best_iter} (position {best_index})")
            print(f"BOHB loss: {best_bohb_loss}")

            # BOHB Losses for each training of this run
            single_training_losses.append(best_bohb_loss)
            # Add model selection losses to the DataFrame
            metrics['run_modsel_losses'] = run_modsel_losses.tolist()
            metrics['bohb_losses'] = bohb_losses.tolist()
            single_training_infos.append(metrics.to_dict(orient='list'))

        # Average BOHB losses for trainings in this run (meaningles if run_n_avg is 1)
        single_training_losses = np.array(single_training_losses)
        bohb_loss = single_training_losses.mean().item()
        var = single_training_losses.var().item()

        print(f"BOHB Losses of {self.run_n_avg} runs: {single_training_losses}")
        print(f"Mean: {bohb_loss}")
        print(f"Variance: {var}")

        print()

        return ({
            'loss': bohb_loss,
            'info': {
                'criterion': str(self.model_criterion),
                'criterion_info': self.model_criterion.info,
                'run_criterion': str(self.run_model_criterion),
                'run_criterion_info': self.run_model_criterion.info,
                'var': var,
                'single_info': single_training_infos
            }
        })


def parse_args():
    parser = argparse.ArgumentParser()

    add_base_args(parser)

    parser.add_argument('--exp-name', default=None, type=str, help="Name of the experiment")
    parser.add_argument('--exp-suffix', default=None, type=str, help="Suffix to experiment name")

    # BOHB
    parser.add_argument('--min-budget', default=2000, type=int, help="Minimum budget (iterations)")
    parser.add_argument('--max-budget', default=6000, type=int, help="Maximum budget (iterations)")
    parser.add_argument('--eta', default=3., type=float, help="Eta value for BOHB")
    parser.add_argument('--run-n-avg', default=1, type=int,
                        help="A configuration should be evaluated training a model N times and averaging the metric")
    parser.add_argument('--criterion', type=ModelCriterion, required=True, help="Model selection criterion for BOHB (across runs)")
    parser.add_argument('--run-criterion', type=ModelCriterion, required=False, default=None, help="Model selection criterion to select best epoch (within training). If not specified is the same as the BOHB criterion")

    # Hyperparameters (the non-optimized ones)
    # (none BOHB-specific)

    # Distributed optimization
    parser.add_argument('--worker', action='store_true', help="Launch this process as a worker")
    parser.add_argument('--master', action='store_true', help="Launch this process as a master")
    parser.add_argument('--num-workers', default=1, type=int, help="Minimum number of parallel optimization worker processes")
    parser.add_argument('--num-iterations', default=24, type=int, help="Number of optimizer iterations")
    parser.add_argument('--nic-name', type=str, help="Network interface")
    parser.add_argument('--loglevel', type=str, default='info', choices=('critical', 'warning', 'info', 'debug'))
    parser.add_argument('--timeout', type=int, default=1200, help="Worker timeout")
    parser.add_argument('--previous', type=str, default='')

    args = parser.parse_args()

    return args


def get_default_exp_name(args):
    def shorten_domain(domain):
        return re.sub(r'-.*', '', domain)

    params = (
        # ABAS and DA method (ALDA or DANN)
        f'abas-{args.da}',
        # Optimization metric (abbreviated. "target_accuracy" -> "ta", "target_calinski_harabasz_score" -> "tchs")
        ''.join(map(lambda s: s[0], re.split(r'[\s_-]+', str(args.criterion)))),
        # Domains ("clipart-oh" is shortened into "clipart")
        "{}-{}".format(shorten_domain(args.source), shorten_domain(args.target)),
        # Backbone
        args.net,
        # Discriminator type
        'fc',
        # Budgets
        args.min_budget,
        args.max_budget,
        # Eta value
        args.eta,
        # Number of iterations
        args.num_iterations
    )

    return "_".join(map(lambda v: str(v), params))


def main():
    args = parse_args()

    # Set log level
    logging.basicConfig(level={
        'critical': logging.CRITICAL,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }[args.loglevel])

    # Name for the current experiment (optimization, not single training)
    exp_name = args.exp_name or get_default_exp_name(args) + (('_' + args.exp_suffix) if args.exp_suffix else '')

    logdir = os.path.join(args.logdir, exp_name)
    shared_dir = os.path.join(logdir, 'master')
    os.makedirs(shared_dir, exist_ok=True)  # Also creates logdir if it does not exist

    host = hpns.nic_name_to_host(args.nic_name)

    # If this is meant to be a worker process, launch it
    if args.worker:
        w = AbasWorker(run_id=exp_name,
                       source=args.source,
                       target=args.target,
                       net=args.net,
                       load_workers=args.load_workers,
                       max_iter=args.max_iter,
                       logdir=args.logdir,
                       ds_root=args.data_root,
                       no_tqdm=args.no_tqdm,
                       gpu=args.gpu,
                       run_n_avg=args.run_n_avg,
                       da_method=args.da,
                       model_criterion=args.criterion,
                       run_model_criterion=args.run_criterion or args.criterion,
                       kill_diverging=args.kill_diverging,

                       host=host,
                       timeout=args.timeout)
        w.load_nameserver_credentials(working_directory=shared_dir)
        w.run(background=False)
        # Nothing to do, exit
        print("Done")
        exit(0)

    # If we are here we expect to be a master
    if not args.master:
        print("Nothing to do (not a master nor a worker process)")
        exit(1)

    # Running as master!

    # Log info
    Logger(logdir=logdir, run_name='master', use_tqdm=False, use_tb=False)

    # Init the nameserver (random port)
    ns = hpns.NameServer(run_id=exp_name, host=host, port=0, working_directory=shared_dir)
    ns_host, ns_port = ns.start()
    print("Nameserver on {}:{}".format(ns_host, ns_port))

    # These hyperparameters are passed through the command line and are not optimized
    hp = {
        'base.lr': args.lr,
        'base.bs': args.bs,
        'base.wd': args.wd,
    }

    # Load previous runs
    previous_res = None
    if args.previous != '':
        if os.path.isdir(args.previous):
            previous_res = hpres.logged_results_to_HBS_result(args.previous)
        else:
            with open(args.previous, 'rb') as fp:
                previous_res = pickle.load(fp)

    # Safe file removal
    remove_file(os.path.join(shared_dir, 'config.json'))
    remove_file(os.path.join(shared_dir, 'results.json'))

    # Launch BOHB
    opt_logger = hpres.json_result_logger(directory=shared_dir, overwrite=False)
    bohb = BOHB(
        configspace=AbasWorker.get_configspace(hp),
        previous_result=previous_res,
        run_id=exp_name,

        min_budget=args.min_budget, max_budget=args.max_budget,
        eta=args.eta,

        host=host,
        nameserver=ns_host, nameserver_port=ns_port,
        ping_interval=15,

        result_logger=opt_logger
    )

    res = bohb.run(n_iterations=args.num_iterations, min_n_workers=args.num_workers)

    # Done
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    # Save results
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    all_runs = res.get_all_runs()

    with open(os.path.join(logdir, 'result_{}.pkl'.format(exp_name)), 'wb') as fp:
        pickle.dump(res, fp)

    print(f"Best found configuration: {id2config[incumbent]['config']}")
    print(f"Total number of sampled unique configurations: {len(id2config.keys())}")
    print(f"Total runs {len(res.get_all_runs())}")
    print("ABAS run took {:.1f} seconds".format(
            all_runs[-1].time_stamps['finished'] - all_runs[0].time_stamps['started']))


if __name__ == '__main__':
    main()
