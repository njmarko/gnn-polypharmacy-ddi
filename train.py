import argparse
import pickle

import wandb

from ddi_dataset import create_ddi_dataloaders


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

    # Directory containing precomputed training data split.
    parser.add_argument('-input_data_path', '--input_data_path', default=None,
                        help="Input data path, e.g. ./data/decagon/")

    parser.add_argument('-f', '--fold', default='1/10', type=str,
                        help="Which fold to test on, format x/total")

    opt = parser.parse_args()

    data_loaders = create_ddi_dataloaders(opt)

    wandb.init(entity=opt.entity, project=opt.project_name, group=opt.group, job_type=opt.job_type, config=opt)


if __name__ == "__main__":
    main()
