from torch.utils.data import Dataset, DataLoader
import json
import requests
from itertools import combinations
import numpy as np
import torch
from pathlib import Path

class Tournament:
    def __init__(self, tournament_id, max_members=9, cache=True):
        self.tournament_id = tournament_id
        self.max_members = max_members
        self.cache = cache
        self.keep = ["team", "questionsTotal", "position", "teamMembers"]
        # JSON with results and team members
        self.json = self.get_json()
        # Number of participating teams
        self.n_teams = len(self.json)

        # Two datasets
        self.matches = Matches(self.json, max_members)
        self.ranking = Ranking(self.json, max_members)

    def get_json(self):
        query = f'http://api.rating.chgk.net/tournaments/{self.tournament_id}/results?' \
                f'includeTeamMembers=1&includeMasksAndControversials=0&includeTeamFlags=0&includeRatingB=0'

        path = f'./data/{self.tournament_id}.json'
        if Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
        else:
            r = requests.get(query)
            if r.status_code == 200:
                output = [{k: d[k] for k in self.keep} for d in r.json()]
                if self.cache:
                    with open(path, 'w+') as f:
                        json.dump(output, f, indent=2)
                return output
            else:
                raise RuntimeError(f'request {query} failed with status code {r.status_code}')

    def get_positions(self):
        return [t['position'] for t in self.json]

    def get_questions(self):
        return [t['questionsTotal'] for t in self.json]


class Matches(Dataset):
    def __init__(self, json, max_members=9):
        self.max_members = max_members
        # JSON with results and team members
        self.json = json
        # Number of participating teams
        self.n_teams = len(self.json)
        # Pairs of teams form matches
        self.pairs = list(combinations(np.arange(self.n_teams), 2))
        # Number of pairs
        self.n_pairs = len(self.pairs)

    def __getitem__(self, index):
        # Get pair by index
        pair = self.pairs[index]
        # Get full team info from tournament json
        teams = [self.json[i] for i in pair]
        # For each team get team members' ids
        ids = [np.array([int(m['player']['id']) for m in t['teamMembers']][:self.max_members]) for t in teams]
        # Get match results (1 if teams[0] wins, -1 if teams[1] wins, 0 if it's a tie)
        questions = [int(t['questionsTotal']) for t in teams]
        if questions[0] > questions[1]:
            result = 1
        elif questions[0] < questions[1]:
            result = -1
        else:
            result = 0

        team_1 = torch.zeros(self.max_members).long()
        team_2 = torch.zeros(self.max_members).long()

        team_1[:len(ids[0])] = torch.from_numpy(ids[0])
        team_2[:len(ids[1])] = torch.from_numpy(ids[1])

        return team_1, team_2, torch.tensor(result)

    def __len__(self):
        return self.n_pairs


def collate_match(batch, max_members=9):
    """
    Collate match to torch.Tensor
    :return: team_1 (torch.Tensor, long, bs x self.max_members),
             team_2 (torch.Tensor, long, bs x self.max_members),
             result (torch.Tensor, long,
    """
    bs = len(batch)
    team_1 = torch.zeros((bs, max_members)).long()
    team_2 = torch.zeros((bs, max_members)).long()
    results = torch.tensor([b[2] for b in batch]).long()

    for i in range(bs):
        team_1[i] = batch[i][0][:max_members]
        team_2[i] = batch[i][1][:max_members]

    return team_1, team_2, results


class Ranking(Dataset):
    def __init__(self, json, max_members=9):
        self.json = json
        self.max_members = max_members
        self.teams = self.get_teams()
        self.n_teams = len(self.teams)

    def __len__(self):
        return self.n_teams

    def __getitem__(self, item):
        team = np.array(self.teams[item][:self.max_members])
        team_t = torch.zeros(self.max_members).long()
        team_t[:len(team)] = torch.from_numpy(team)
        return team_t

    def get_teams(self):
        """
        Get all the teams with members
        :return: list
        """
        return [[int(m['player']['id']) for m in t['teamMembers']] for t in self.json]


def collate_teams(batch, max_members=9):
    """
    Collate all teams in tournament for scoring
    :return: teams (torch.Tensor, long, bs x self.max_members))
    """
    bs = len(batch)
    teams = torch.zeros((bs, max_members)).long()
    for i in range(bs):
        teams[i] = batch[i][:max_members]
    return teams
