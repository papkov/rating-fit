import torch
import pandas as pd
import requests
from tqdm import tqdm
from scipy.stats import spearmanr


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


def get_baseline_correlation(tournaments_path='./data/tournaments.csv'):
    tournaments = pd.read_csv(tournaments_path)
    query = "http://api.rating.chgk.net/tournaments/{}/results?" \
            "includeTeamMembers=0&includeMasksAndControversials=0&includeTeamFlags=0&includeRatingB=1"
    correlations = []
    for tid in tqdm(tournaments['id'], position=0):
        try:
            r = requests.get(query.format(tid)).json()
            positions = [t['position'] for t in r]
            predicted = [t['rating']['predictedPosition'] for t in r]
            correlations.append({'id': tid, 'corr': spearmanr(positions, predicted)[0]})
        except Exception as e:
            print(f'Error in id {tid}:', e)

    return pd.DataFrame(correlations)
