import argparse
import os
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.testing._internal.common_quantization import AverageMeter
import torch.optim as optim
from torch_geometric.utils import degree
from tqdm import tqdm
import wandb
import logging

from ddi_dataset import create_ddi_dataloaders
from model.GNNModel import GraphTransformer


def main():
    parser = argparse.ArgumentParser()

    # Wandb logging options
    parser.add_argument('-entity', '--entity', type=str, default="neural-networks",
                        help="Name of the team. Multiple projects can exist for the same team.")
    parser.add_argument('-project_name', '--project_name', type=str, default="gnn-polypharmacy-ddi",
                        help="Name of the project. Each experiment in the project will be logged separately"
                             " as a group")
    parser.add_argument('-group', '--group', type=str, default="default_experiment",
                        help="Name of the experiment group. Each model in the experiment group will be logged "
                             "separately under a different type.")
    parser.add_argument('-save_model_wandb', '--save_model_wandb', type=bool, default=True,
                        help="Save best model to wandb run.")
    parser.add_argument('-job_type', '--job_type', type=str, default="train",
                        help="Job type {train, eval}.")

    # Dataset

    parser.add_argument('-dataset', '--dataset', metavar='D', type=str.lower,
                        choices=['decagon'],
                        help='Name of dataset to used for training [DECAGON]')
    parser.add_argument('-nr', '--train_neg_pos_ratio', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=100)

    # Training options
    parser.add_argument('-device', '--device', type=str, default='cuda', help="Device to be used")
    parser.add_argument('-e', '--n_epochs', type=int, default=10000, help="Max number of epochs")

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)

    parser.add_argument('-l2', '--l2_lambda', type=float, default=0)
    parser.add_argument('-drop', '--dropout', type=float, default=0.1)
    parser.add_argument('-global_step', '--global_step', type=int, default=0)

    # Directory containing precomputed training data split.
    parser.add_argument('-input_data_path', '--input_data_path', default=None,
                        help="Input data path, e.g. ./data/decagon/")

    parser.add_argument('-f', '--fold', default='1/10', type=str,
                        help="Which fold to test on, format x/total")

    opt = parser.parse_args()
    opt.device = 'cuda' if torch.cuda.is_available() and (opt.device == 'cuda') else 'cpu'
    print(opt.device)

    train_loader, val_loader = create_ddi_dataloaders(opt)

    # Computing degree for PNAConv https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
    # Compute the maximum in-degree in the training data.
    # max_degree = -1
    # for data in train_loader.dataset:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     max_degree = max(max_degree, int(d.max()))
    #
    # # Compute the in-degree histogram tensor
    # deg = torch.zeros(max_degree + 1, dtype=torch.long)
    # for data in train_loader.dataset:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    deg = torch.ones(30, dtype=torch.long)

    model = GraphTransformer(
        batch_size=opt.batch_size,
        num_atom_type=100,
        deg=deg
    ).to(opt.device)

    optimizer = optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_lambda)

    wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group, job_type=opt.job_type, config=opt)

    best_val = 0
    averaged_model = model.state_dict()
    best_val_auroc = -np.inf
    best_model_path = None
    for epoch in range(opt.n_epochs):
        train_loss, epoch_time, averaged_model = train(model, train_loader, optimizer, averaged_model, opt)
        logging.info(f"  Train loss: {train_loss}, time: {epoch_time}")
        training_model = model.state_dict()

        # Using average model for validation
        model.load_state_dict(averaged_model)

        val_metrics, val_time = validate(model, val_loader, opt)

        wandb.log({"validation_performance": val_metrics})

        logging.info(f"  Validation: {val_metrics['auroc']:.4f}, time: {val_time}")

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            Path(f'experiments/{opt.group}').mkdir(exist_ok=True)
            new_best_path = os.path.join(f'experiments/{opt.group}',
                                         f'train-{opt.group}-epoch{epoch}'
                                         f'-metric{val_metrics["auroc"]:.4f}.pt')
            torch.save({'global_step': opt.global_step,
                        'model': averaged_model,
                        'threshold': val_metrics['threshold']}, new_best_path)
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = new_best_path

        model.load_state_dict(training_model)


def train(model, data_loader, optimizer, averaged_model, opt):
    model.train()
    start_time = timeit.default_timer()

    avg_training_loss = AverageMeter("Train epoch loss avg")

    for batch in tqdm(data_loader, mininterval=5, desc="Training"):
        optimizer.zero_grad()

        # Custom Loss update
        lr = opt.learning_rate * (0.96 ** (opt.global_step / 1000000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        pos_batch, neg_batch, seg_pos_neg = batch
        pos_batch = [v.to(opt.device) for v in pos_batch]
        neg_batch = [v.to(opt.device) for v in neg_batch]
        seg_pos_neg = seg_pos_neg.to(opt.device)

        predictions_pos = model(*pos_batch)
        predictions_neg = model(*neg_batch)
        loss = max_margin_loss_fn(predictions_pos, predictions_neg, seg_pos_neg)

        loss.backward()
        optimizer.step()

        sz_b = seg_pos_neg.size(0)
        avg_training_loss.update(loss.detach(), sz_b)

        # Custom average model formula
        for var in model.state_dict():
            averaged_model[var] = 0.9 * averaged_model[var] + (1 - 0.9) * model.state_dict()[var]
        opt.global_step += 1
        wandb.log({"training_loss": loss.detach()})

    epoch_time = timeit.default_timer() - start_time

    return avg_training_loss.avg, epoch_time, averaged_model


def validate(model, data_loader, opt):
    model.eval()
    score, label, seidx = [], [], []
    start_time = timeit.default_timer()
    with torch.no_grad():
        for batch in tqdm(data_loader, mininterval=3, desc='Validation'):
            *batch, batch_label = batch
            batch = [v.to(opt.device) for v in batch]  # move to GPU if needed

            batch_score = model(*batch)
            if batch_score is None:
                break

            label += [batch_label]
            score += [batch_score]
            seidx += [batch[-2]]

    label = np.hstack(label)
    score = np.hstack([s.cpu() for s in score])
    seidx = np.hstack([s.cpu() for s in seidx])

    threshold = get_optimal_thresholds_for_rels(seidx, label, score)
    instance_threshold = threshold[seidx]

    pred = score > instance_threshold

    performance = {
        'auroc': metrics.roc_auc_score(label, score),
        'avg_p': metrics.average_precision_score(label, score),
        'f1': metrics.f1_score(label, pred, average='binary'),
        'p': metrics.precision_score(label, pred, average='binary'),
        'r': metrics.recall_score(label, pred, average='binary'),
        'threshold': threshold
    }
    epoch_time = timeit.default_timer() - start_time
    return performance, epoch_time


def max_margin_loss_fn(pos_eg_score, neg_eg_score, seg_pos_neg, margin=1):
    pos_eg_score = pos_eg_score.index_select(0, seg_pos_neg)
    return torch.mean(F.relu(margin - pos_eg_score + neg_eg_score))


# https://arxiv.org/abs/1905.005342
def get_optimal_thresholds_for_rels(relations, goal, score, interval=0.01):
    def get_optimal_threshold(goal, score):
        """ Get the threshold with maximized accuracy"""
        if (np.max(score) - np.min(score)) < interval:
            optimal_threshold = np.max(score)
        else:
            thresholds = np.arange(np.min(score), np.max(score), interval).reshape(1, -1)
            score = score.reshape(-1, 1)
            goal = goal.reshape(-1, 1)
            optimal_threshold_idx = np.sum((score > thresholds) == goal, 0).argmax()
            optimal_threshold = thresholds.reshape(-1)[optimal_threshold_idx]
        return optimal_threshold

    unique_rels = np.unique(relations)
    rel_thresholds = np.zeros(int(unique_rels.max()) + 1)

    for rel_idx in unique_rels:
        rel_mask = np.where(relations == rel_idx)
        rel_goal = goal[rel_mask]
        rel_score = score[rel_mask]
        rel_thresholds[rel_idx] = get_optimal_threshold(rel_goal, rel_score)

    return rel_thresholds


if __name__ == "__main__":
    main()
