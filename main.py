import argparse
from trainer import Trainer
import requests
import datetime
import pandas as pd

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_tournaments(date_start='2012-03-16', date_end='2020-03-16'):

    query = f'http://api.rating.chgk.net/tournaments?' \
            f'dateStart[after]={date_start}&dateEnd[strictly_before]={date_end}&page='
    get_json = lambda x: requests.get(x).json()

    page = 1
    r = get_json(query + str(page))
    fields = ['id', 'dateStart', 'dateEnd']
    tournaments = [{f: t[f] for f in fields} for t in r if isinstance(t, dict)]
    while r:
        page += 1
        r = get_json(query + str(page))
        tournaments.extend([{f: t[f] for f in fields} for t in r if isinstance(t, dict)])
        print(f'Reading tournaments page {page:04d}', end='\r')

    return pd.DataFrame(tournaments)


def parse_args():
    parser = argparse.ArgumentParser(description='Model training for player ratings')
    return parser.parse_args()


def main():
    args = parse_args()

    tournaments = pd.read_csv('./data/tournaments.csv')
    trainer = Trainer(tournament_list=tournaments['id'])
    trainer.fit()


if __name__ == '__main__':
    main()
