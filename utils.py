import torch
import pandas as pd
import requests


def embeddings_to_df(path='./checkpoints/checkpoint.pth'):
    rating = torch.load(path)['model']['emb.weight'].cpu().numpy().flatten()
    rating = pd.DataFrame(dict(rating=rating)).sort_values('rating', ascending=False)
    return rating


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
