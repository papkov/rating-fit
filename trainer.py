import torch
from dataset import *
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd


class Trainer:
    def __init__(self, name='model',
                 lr=0.1, model=None, optimizer=None, tournament_list=[],
                 clip_zero=True,
                 checkpoint_path='./checkpoints', save_each=False, cache=True,
                 jobs=4, batch_size=512):
        # Checkpoints
        self.name = name
        self.checkpoint_path = checkpoint_path
        self.save_each = save_each
        self.history = []
        self.cache = cache

        # Tournaments
        self.tournament_list = tournament_list
        self.total = len(self.tournament_list)

        # Model
        self.model = Model() if model is None else model
        # Default lr and other optimization parameters
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-8) \
            if optimizer is None else optimizer
        self.clip_zero = clip_zero

        # CUDA usage
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        # For DataLoaders
        self.jobs = jobs
        self.bs = batch_size

    def one_epoch(self, tournament_id: int, epoch=0):
        """
        One epoch is a single tournament here
        :return:
        """

        # TODO: tournament pre-fetcher
        tournament = Tournament(tournament_id, cache=self.cache)

        # Measure correlation before to see whether gradient update took effect
        correlation_before = self.get_prediction_correlation(tournament)
        correlation_after = 0

        # Prepare Trainer
        self.model.train()
        # For optimizer, keep embedding LR the same, but scale head LR by number of teams (more teams -> larger LR)
        # self.optimizer.lr = self.optimizer.lr * something
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

            # This condition is needed to update tqdm
            if i == (len(dl_match) - 1):
                # Perform optimizer step once in an epoch (we consider all the matches simultaneous)
                self.optimizer.step()
                # Clip weights if necessary
                if self.clip_zero:
                    self.model.emb.apply(self.model.clipper)
                # Scale head so the output would always be a weighted average
                with torch.no_grad():
                    self.model.head.weight.div_(torch.sum(self.model.head.weight))
                    # self.model.head.weight = torch.nn.Parameter(self.model.head.weight /
                    #                                             torch.sum(self.model.head.weight), requires_grad=True)

                # Print difference in correlation
                correlation_after = self.get_prediction_correlation(tournament)
                postfix = {'loss': f'{cum_loss / (len(dl_match) + 1):.4f}',
                           'corr': f'{correlation_before:.4f} -> {correlation_after:.4f}',
                           }

            else:
                postfix = {'loss': f'{cum_loss / (i + 1):.4f}'}

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
            try:
                loss, correlation_before, correlation_after = self.one_epoch(tournament_id, epoch)
                self.save_checkpoint(epoch=epoch,
                                     tournament_id=tournament_id,
                                     loss=loss,
                                     correlation_before=correlation_before,
                                     correlation_after=correlation_after,
                                     )
            except KeyboardInterrupt:
                print('\nInterrupted')
                break
            except Exception as e:
                print(f'\nError occurred for epoch {epoch} id{tournament_id}\n{e}')
                continue

    def save_checkpoint(self, **kwargs):
        # Save history
        self.history.append(kwargs)
        pd.DataFrame(self.history).to_csv(f'{self.checkpoint_path}/{self.name}.csv')

        # Save state
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        state.update(kwargs)
        name = self.name
        if self.save_each:
            name += '_'.join([f'{k}={v}' for k, v in kwargs.items()])
        torch.save(state, f'{self.checkpoint_path}/{name}.pth')
