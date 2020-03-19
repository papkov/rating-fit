import unittest
from dataset import *
from model import *
from trainer import *
import torch


class TestCase(unittest.TestCase):

    def test_tournament(self, tournament_id=5923):
        t = Tournament(5923)
        print(f'\nLoaded tournament {tournament_id}: {t.n_teams} teams, {t.matches.n_pairs} pairs')
        print(f'Pair example: {t.matches[0]}')
        print(f'Team example: {t.ranking[0]}')

    def test_model(self):
        m = Model()
        team_1 = torch.tensor([1, 2, 3, 4, 5]).long()[None, ...]
        team_2 = torch.tensor([1, 2, 3, 4, 5]).long()[None, ...]
        result = torch.tensor([0]).long()
        loss = m(team_1, team_2, result)
        # Expected loss
        assert torch.isclose(loss, torch.tensor([[0.6931]]), atol=1e-03)
        print(f'\nLoss {loss}')
        torch.save(m, './checkpoints/tests.pth')

    def test_trainer(self, repeats=3, tournament_id=5923):
        """
        We expect to see increase in correlation between predictions and results
        :param repeats:
        :param tournament_id: https://rating.chgk.info/tournament/6220
        :return:
        """
        trainer = Trainer()
        for r in range(repeats):
            trainer.one_epoch(tournament_id)
            # print(trainer.model.head.weight)

    def test_get_questions(self):
        t = Tournament(5923)
        print(t.get_questions())

    def test_get_scores(self):
        tournament = Tournament(5923)
        trainer = Trainer()
        print(trainer.get_scores(tournament))


if __name__ == '__main__':
    unittest.main()
