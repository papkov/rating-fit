import torch
from dataset import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr


class Trainer:
    def __init__(self, lr=0.5, model=None, tournament_list=[]):
        self.model = Model() if model is None else model
        self.lr = lr
        # self.optimizer = torch.optim.SGD([
        #     {'params': self.model.emb.parameters()},
        #     {'params': self.model.head.parameters(), 'lr': self.lr}
        # ], lr=self.lr, momentum=0.9)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.)

        # CUDA usage
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        self.jobs = 4
        self.bs = 64
        self.tournament_list = tournament_list
        self.total = len(self.tournament_list)
        self.checkpoint_path = './checkpoints/'

    def one_epoch(self, tournament_id: int, epoch=0):
        """
        One epoch is a single tournament here
        :return:
        """

        # TODO: tournament pre-fetcher
        tournament = Tournament(tournament_id)

        # Measure correlation before to see whether gradient update took effect
        correlation_before = self.get_prediction_correlation(tournament)

        # Prepare Trainer
        self.model.train()
        # For optimizer, keep embedding LR the same, but scale head LR by number of teams (more teams -> larger LR)
        # self.optimizer = torch.optim.SGD([
        #     {'params': self.model.emb.parameters(), 'lr': self.lr},
        #     {'params': self.model.head.parameters(), 'lr': min(self.lr, self.lr * tournament.n_teams / 100)}
        # ], lr=self.lr, momentum=0.9)
        self.optimizer.zero_grad()

        # collate_fn = lambda x: collate_match(x, tournament.max_members)
        dl_match = DataLoader(tournament.matches, num_workers=self.jobs, batch_size=self.bs, shuffle=True)

        iterator = tqdm(dl_match, position=0, desc=f'epoch {epoch+1:04d}/{self.total} id{tournament_id}')
        cum_loss = 0
        for i, (team_1, team_2, result) in enumerate(iterator):
            # Calculate the loss based on match results
            loss = self.model(team_1.to(self.device), team_2.to(self.device), result.to(self.device))
            # Scale the loss by number of updates per team
            # loss /= (tournament.matches.n_pairs - 1)
            # Do backward step, accumulate loss and gradients
            loss.backward()
            cum_loss += loss.item()
            postfix = {'loss': f'{cum_loss / (i+1):.4f}'}
            iterator.set_postfix(postfix)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i == (len(dl_match) - 1):
                # Perform optimizer step once in an epoch
                # (otherwise predictions will be affected -- we consider all the matches simultaneous)

                # Print difference in correlation
                correlation_after = self.get_prediction_correlation(tournament)
                postfix = {'loss': f'{cum_loss / (len(dl_match) + 1):.4f}',
                           'corr': f'{correlation_before:.4f} -> {correlation_after:.4f}',
                           }
                iterator.set_postfix(postfix)

        return cum_loss / len(dl_match), correlation_before, correlation_after

    @torch.no_grad()
    def get_scores(self, tournament: Tournament):
        """
        Get scores for all the teams
        :param tournament:
        :return: np.array, np.float32
        """
        self.model.eval()
        # collate_fn = lambda x: collate_teams(x, tournament.max_members)
        dl_rank = DataLoader(tournament.ranking, num_workers=self.jobs, batch_size=self.bs, shuffle=False)
        iterator = tqdm(dl_rank, position=0, desc=f'{tournament.tournament_id} ranking', disable=True)
        scores = []
        for i, team in enumerate(iterator):
            score = self.model.get_team_score(team.to(self.device))
            scores.append(score.cpu().numpy())

        scores = np.concatenate(scores)
        return scores.flatten()

    def get_prediction_correlation(self, tournament: Tournament):
        scores = self.get_scores(tournament)
        # Calculate correlation with total questions
        # return spearmanr(scores, tournament.get_questions())[0]
        return spearmanr(scores, tournament.get_positions()[::-1])[0]

    def fit(self):
        for epoch, tournament_id in enumerate(self.tournament_list):
            loss, correlation_before, correlation_after = self.one_epoch(tournament_id, epoch)
            self.save_checkpoint(loss=loss,
                                 correlation_before=correlation_before,
                                 correlation_after=correlation_after,
                                 epoch=epoch)

    def save_checkpoint(self, name='checkpoint', **kwargs):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        state.update(kwargs)

        path = f'{self.checkpoint_path}/{name}.pth'
        torch.save(state, path)
