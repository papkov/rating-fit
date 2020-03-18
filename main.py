import argparse
from trainer import Trainer
from model import Model
from utils import *
import torch
import requests
import datetime
import pandas as pd
from pathlib import Path
import json

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(prog='rating',
                                     description='Model training for player ratings')
    # Checkpoints
    parser.add_argument('--name', '-n', default='model',
                        help='Model name to save checkpoint')
    parser.add_argument('--checkpoint-path', default='./checkpoints/',
                        help='Dir to save checkpoints to')
    parser.add_argument('--save-each', action='store_true', default=False,
                        help='Save each epoch (tournament) checkpoint in separate file')
    # Tournaments to use within certain date range
    parser.add_argument('--date-start', '-s', default='2012-03-16',
                        help='Use tournaments starting this date (format as 2012-03-16)')
    parser.add_argument('--date-end', '-e', default='2020-03-16',
                        help='Use tournaments until this date (format as 2020-03-16)')
    parser.add_argument('--tournaments', default='./data/tournaments.csv',
                        help='Use preloaded list of tournaments')
    parser.add_argument('--cache', default=True,
                        help='Save tournament data')
    # Optimization parameters
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--wd', default=1e-8, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    # Training parameters
    parser.add_argument('--loss', choices=['logsigmoid', 'sigmoid'], default='sigmoid',
                        help='Loss function to train a model')
    parser.add_argument('--clip-zero', action='store_true', default=True,
                        help='Shift model embeddings so that min == 0')
    parser.add_argument('--batch-size', '-b', default=512, type=int,
                        help='Batch size for data loaders')
    parser.add_argument('--workers', '-j', default=4, type=int,
                        help='Number of parallel workers in data loaders')
    parser.add_argument('--take_best', default=6, type=int,
                        help='Number of best players taken into account')

    return parser.parse_args()


def save_args(args):
    with open(f'{args.checkpoint_path}/{args.name}.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def print_args(args):
    print('Run with parameters:')
    for k, v in args.__dict__.items():
        print(f'{k} = {v}')


def download_tournaments(args):
    try:
        tournaments = pd.read_csv(args.tournaments)
    except FileNotFoundError:
        print(f'Tournaments file not fount at {args.tournaments}\n'
              f'Downloading tournaments from {args.date_start} until {args.date_end}')
        tournaments = get_tournaments(args.date_start, args.date_end)
        Path('./data').mkdir(parents=True, exist_ok=True)
        tournaments.to_csv(f'./data/tournaments_{args.date_start}_{args.date_end}.csv')
    return tournaments


def main():
    # Parse and cache args under the model name for reproducibility
    args = parse_args()
    print_args(args)

    # Make paths
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path('./data').mkdir(parents=True, exist_ok=True)
    save_args(args)

    # Download list of tournaments
    tournaments = download_tournaments(args)

    # Fit the model
    model = Model(loss=args.loss, take_best=args.take_best)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    trainer = Trainer(name=args.name, model=model, optimizer=optimizer, cache=args.cache,
                      tournament_list=tournaments['id'], save_each=args.save_each,
                      jobs=args.workers, batch_size=args.batch_size)
    trainer.fit()


if __name__ == '__main__':
    main()
