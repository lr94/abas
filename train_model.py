#!/usr/bin/env python3
import argparse
import pickle
import time

import pandas as pd
import torch.nn as nn
import torch.optim as opt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import loss
from datasets import *
from logger import Logger
from net import resnet
from model_selection import *
from utils import *


def get_default_run_name(args):
    def shorten(domain):
        return re.sub(r'-.*', '', domain)

    params = (
        args.da,
        "{}-{}".format(shorten(args.source), shorten(args.target)),
        args.net,
        'fc',
        args.max_iter,
        args.bs,
        args.lr
    )

    return "_".join(map(lambda v: str(v), params))


def test(loader, model, device, source_feats=None):
    tot_correct = 0
    tot = 0
    model.eval()  # just to be sure
    feat_size = model.output_size()

    num_classes = get_class_count(loader.dataset)

    feats = torch.empty(len(loader.dataset), feat_size)
    pseudo_labels = torch.zeros(len(loader.dataset), dtype=torch.int64)
    all_labels = torch.zeros(len(loader.dataset), dtype=torch.int64)
    all_outputs = torch.empty(len(loader.dataset), num_classes)

    entropy = 0
    class_loss = 0
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = map_to_device(device, (inputs, labels))
            outputs, f = model(inputs)

            # save features for later
            s = inputs.shape[0]
            feats[tot:tot + s] = f.cpu()
            pseudo_labels[tot:tot + s] = outputs.argmax(1).cpu()
            all_outputs[tot:tot + s] = outputs.cpu()
            all_labels[tot:tot + s] = labels.cpu()

            # entropy, accuracy and class loss
            preds = torch.argmax(outputs, dim=1)
            entropy += loss.entropy_loss(outputs).item()
            class_loss += ce_loss(outputs, labels).item() * inputs.size(0)

            tot_correct += torch.sum(preds == labels).item()
            tot += s

    # Diversity loss (computed on full dataset)
    pb_pred_tgt = torch.nn.functional.softmax(all_outputs, dim=1).sum(dim=0)
    pb_pred_tgt = pb_pred_tgt / pb_pred_tgt.sum()
    target_div_loss_full = - torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6))).item()
    target_div_loss_full /= num_classes

    feats = feats.numpy()
    pseudo_labels = pseudo_labels.numpy()
    sil, cal = get_clustering_performance(feats, pseudo_labels, num_classes, source_feats=source_feats)
    accuracy = tot_correct / tot
    entropy /= tot
    class_loss /= tot
    metrics = {
        'accuracy': accuracy,
        'entropy_loss': entropy,

        # Diversity loss (on full dataset as one big batch)
        'div_loss': target_div_loss_full,

        'class_loss': class_loss,
        'silhouette_score': sil,
        'calinski_harabasz_score': cal
    }
    return metrics, pseudo_labels, feats


def run_training(
        source,
        target,
        dataset_root,
        net_name,
        da_method,
        max_iter,
        stop_iter,
        test_iter,
        logdir,
        run_name,
        gpu_id,
        load_workers,
        config,
        test_src: bool = False,
        use_tqdm: bool = True,
        kill_diverging: bool = False):
    dev = torch.device(f'cuda:{gpu_id}')

    if kill_diverging:
        assert test_src

    # Get config
    # Config arrives here (from BOHB or direct cli invocation) as a dictionary like
    # {'disc.dropout': 0.5, 'net.bottleneck_size_log': 9}
    # We separate it in something like
    # {'disc': {'dropout': 0.5'}, 'net': {'bottleneck_size_log': 9}}
    config = split_dict(config)
    # Disc args are not meaningful without DA
    if da_method != 'so':
        # Default disc args
        disc_args = {
            'dropout': 0.5,
            'num_fc_layers': 3,
            'hidden_size_log': 10
        }
        # Update with the ones coming from config (if any)
        disc_args.update(config.get('disc', {}))
        # Some args might be defined as log2. Replace them (bottleneck_size_log -> bottleneck_size)
        remove_log_hps(disc_args)
        # Print disc args
        print(f"Discriminator config: {disc_args}")
    # Very similar, but for the backbone
    net_args = {
        'use_bottleneck': da_method != 'so',
        'bottleneck_size_log': 9
    }
    net_args.update(config.get('net', {}))
    remove_log_hps(net_args)
    print(f"Backbone config: {net_args}")
    # Now net_args and disc_args are ready to be passed to the network constructors as **kwargs :)
    bs, lr, wd = config['base']['bs'], config['base']['lr'], config['base']['wd']

    # Load datasets and their number o classes
    dset_src_train, dset_src_test, dset_trg_train, dset_trg_test, num_classes = \
        prepare_datasets(source, target, dataset_root)

    dload_src_train = DataLoader(dset_src_train, batch_size=bs, shuffle=True, num_workers=load_workers, drop_last=True)
    dload_src_test = DataLoader(dset_src_test, batch_size=bs, shuffle=True, num_workers=load_workers)
    dload_trg_train = DataLoader(dset_trg_train, batch_size=bs, shuffle=True, num_workers=load_workers, drop_last=True)
    dload_trg_test = DataLoader(dset_trg_test, batch_size=bs, shuffle=True, num_workers=load_workers)

    print(f"Source samples: {len(dset_src_train)}")
    print(f"Target samples: {len(dset_trg_train)}")
    print(f"Num classes: {num_classes}")

    # Build network
    base_network = resnet.ResNetFc(
        resnet_name=net_name,
        num_classes=num_classes,
        plug_position=7,
        **net_args
    ).to(dev)
    params = base_network.get_parameters(lr, wd)
    # Source only has no secondary branches
    if da_method != 'so':
        disc_classes = {
            # ( -> confusion matrix)
            'alda': num_classes,
            # ( -> binary domain classifier)
            'dann': 2
        }[da_method]
        discriminator = resnet.Discriminator(in_feature=base_network.output_size(), num_classes=disc_classes,
                                             **disc_args).to(dev)
        params += discriminator.get_parameters(lr, wd)

    # Define optimizer
    optimizer = opt.SGD(
        params=params,
        lr=lr,
        momentum=0.9,
        weight_decay=wd,
        nesterov=True
    )

    # Lr policy
    lr_schedule = LambdaLR(optimizer, lr_lambda=lambda it: (1 + 0.001 * it) ** (-0.75))

    # Logger
    writer = Logger(logdir=logdir, run_name=run_name, use_tb=True, use_tqdm=use_tqdm)

    # Classification loss
    ce_loss = nn.CrossEntropyLoss()

    # Train loop
    len_train_source = len(dload_src_train)
    len_train_target = len(dload_trg_train)
    lambda_val = 0.

    # We store all the metrics here
    metrics = []

    all_pseudolabels = []

    with writer.progress(total=stop_iter, desc="Training") as pb:
        for i in range(stop_iter):
            if (i + 1) % test_iter == 0:
                print(f"Iteration: {i + 1} / {stop_iter} (max: {max_iter})")
                print("Testing...")
                base_network.train(False)
                # This dict contains metric-name -> value pairs for the current epoch
                new_metrics = {}
                if test_src:
                    test_result, _, src_test_feats = test(dload_src_test, base_network, device=dev)
                    # Print accuracy
                    print("Source accuracy: {:.3f} %".format(test_result['accuracy'] * 100))
                    # Add the source metrics to the dict (with the source_ prefix)
                    new_metrics.update({f'source_{k}': v for k, v in test_result.items()})

                test_result, epoch_pseudolabels, _ = test(dload_trg_test, base_network, device=dev,
                                                          source_feats=src_test_feats)
                all_pseudolabels.append(epoch_pseudolabels)
                print(f"Target accuracy: {test_result['accuracy'] * 100:.3f} %")

                writer.add_scalar('train/base_lr', lr_schedule.get_last_lr()[0], i)
                writer.add_scalar('train/lambda', lambda_val, i)

                new_metrics.update({f'target_{k}': v for k, v in test_result.items()})

                # Add all the new metrics to tensorboard logs
                add_scalars(writer, new_metrics, global_step=i, prefix='test/')
                # Add a column with iteration number
                new_metrics.update({'iter': i})
                # Concatenate to older epoch metrics
                metrics.append(new_metrics)

                # Kill this training if source loss goes too high
                if kill_diverging and new_metrics['source_class_loss'] > SOURCE_LOSS_THRESHOLD:
                    if len(metrics) > 0 and new_metrics['source_class_loss'] > metrics[-1]['source_class_loss']:
                        print(f"Increasing source_class_loss exceeds maximum allowed source loss ({new_metrics['source_class_loss']} > {SOURCE_LOSS_THRESHOLD})")
                        break

            # Train one iteration
            base_network.train(True)
            if da_method != 'so':
                discriminator.train(True)

            optimizer.zero_grad()

            # Reset data loops if required
            if i % len_train_source == 0:
                iter_source = iter(dload_src_train)
            if i % len_train_target == 0:
                iter_target = iter(dload_trg_train)

            # Load source
            inputs_source, labels_source = iter_source.next()
            inputs_source, labels_source = map_to_device(dev, (inputs_source, labels_source))

            # Compute source features and classification output
            outputs_source, features_source = base_network(inputs_source)

            # Classification loss
            classifier_loss = ce_loss(outputs_source, labels_source)

            # Actual DA part
            if da_method != 'so':
                # Load target samples without target labels
                inputs_target, _ = iter_target.next()
                inputs_target = inputs_target.to(dev)

                # Compute target features and classification output
                outputs_target, features_target = base_network(inputs_target)

                # Source and target features
                features = torch.cat((features_source, features_target), dim=0)
                # Source and target classification outputs (-> softmax)
                outputs = torch.cat((outputs_source, outputs_target), dim=0)
                softmax_out = nn.Softmax(dim=1)(outputs)

                # CORE
                if da_method == 'dann':
                    p = float(i / max_iter)
                    lambda_val = 2. / (1 + np.exp(-10 * p)) - 1
                    ad_out = discriminator(features, lambda_val)
                    adv_loss = loss.DANN_loss(ad_out)
                    transfer_loss = adv_loss
                    if (i + 1) % test_iter == 0:
                        print("Transfer loss: {:.3f}".format(transfer_loss.item()))
                elif da_method == 'alda':
                    p = float(i / max_iter)
                    lambda_val = 2. / (1 + np.exp(-10 * p)) - 1
                    ad_out = discriminator(features, lambda_val)
                    adv_loss, reg_loss, correct_loss = loss.ALDA_loss(ad_out, labels_source, softmax_out, threshold=0.9)

                    transfer_loss = adv_loss + lambda_val * correct_loss
                    if (i + 1) % test_iter == 0:
                        print("Transfer loss: {:.3f}, reg loss  {:.3f}%".format(transfer_loss.item(),
                                                                                reg_loss.item()))
                    # Backpropagate reg_loss only through the discriminator
                    with base_network.freeze():
                        reg_loss.backward(retain_graph=True)
                # END CORE
            else:
                transfer_loss = 0

            total_loss = classifier_loss + config['base']['weight_da'] * transfer_loss
            total_loss.backward()

            optimizer.step()
            lr_schedule.step()

            if (i + 1) % test_iter == 0 and da_method != 'so':
                writer.add_scalar('train/transfer_loss', transfer_loss.item(), i)
            pb.update(1)

    # Convert list of dicts to dataframe containing metrics
    metrics = pd.DataFrame(metrics)

    # Compute global-pseudolabel accuracy
    all_pseudolabels = np.array(all_pseudolabels)
    global_pseudolabels = compute_time_consistent_pseudolabels(all_pseudolabels, num_classes)
    pseudolabel_acc = np.equal(all_pseudolabels, global_pseudolabels).sum(axis=1) / global_pseudolabels.shape[0]
    # Add it to the metrics dataframe
    metrics['target_pseudolabels'] = pseudolabel_acc

    # Save the metrics
    with open(os.path.join(logdir, run_name, "metrics.pkl"), "wb") as fp:
        pickle.dump(metrics, fp)

    # Log global pseudolabel accuracy to tensorboard
    for i in range(len(all_pseudolabels)):
        writer.add_scalar('test/target_pseudolabels', float(pseudolabel_acc[i]), i * test_iter)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    add_base_args(parser)
    parser.add_argument('--stop-iter', type=int, default=None, help="Stop after n iterations")
    parser.add_argument('--run-name', type=str, default=None, help="Experiment name")
    parser.add_argument('--run-suffix', type=str, default='')
    parser.add_argument('--weight-da', type=float, default=1., help="Weight for Domain Adaptation loss")

    args = parser.parse_args()

    args.stop_iter = args.stop_iter or args.max_iter

    run_name = (args.run_name or get_default_run_name(args)) + (
        ('_' + args.run_suffix) if args.run_suffix != '' else '')
    print("Starting {}".format(run_name))

    print(f"{args.source} -> {args.target}")
    print_args(args, (
        'net',
        'bs',
        'lr',
        'wd',
        'stop_iter',
        'max_iter',
        'gpu'
    ))

    config = {
        'base.bs': args.bs,
        'base.lr': args.lr,
        'base.wd': args.wd,
        'base.weight_da': args.weight_da,
    }
    config.update(args.config)

    run_training(
        source=args.source,
        target=args.target,
        dataset_root=args.data_root,
        net_name=args.net,
        da_method=args.da,
        config=config,
        max_iter=args.max_iter,
        stop_iter=args.stop_iter,
        test_iter=args.test_iter,
        logdir=args.logdir,
        run_name=run_name,
        gpu_id=args.gpu,
        load_workers=args.load_workers,

        use_tqdm=not args.no_tqdm,
        test_src=not args.no_test_source,
        kill_diverging=args.kill_diverging
    )


if __name__ == "__main__":
    main()
