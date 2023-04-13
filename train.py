import argparse
import timeit

import torch
import torch.nn.functional as F
from torch.testing._internal.common_quantization import AverageMeter
import torch.optim as optim
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

    model = GraphTransformer(
        batch_size=opt.batch_size,
        num_atom_type=100
    ).to(opt.device)

    optimizer = optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_lambda)

    wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group, job_type=opt.job_type, config=opt)

    best_val = 0
    averaged_model = None
    for epoch in range(opt.n_epochs):
        train_loss, epoch_time, averaged_model = train(model, train_loader, optimizer, averaged_model, opt)


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

        opt.global_step += 1
        wandb.log({"training_loss": loss.detach()})

    epoch_time = timeit.default_timer() - start_time

    return avg_training_loss.avg, epoch_time, averaged_model


def max_margin_loss_fn(pos_eg_score, neg_eg_score, seg_pos_neg, margin=1):
    pos_eg_score = pos_eg_score.index_select(0, seg_pos_neg)
    return torch.mean(F.relu(margin - pos_eg_score + neg_eg_score))


if __name__ == "__main__":
    main()
