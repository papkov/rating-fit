from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, embedding_dim=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        # embedding layer is sparse, it maps a player to a vector
        self.emb = nn.Embedding(num_embeddings=300000, embedding_dim=embedding_dim, padding_idx=0)
        # on top of embedding we train a simple linear regression
        self.neck = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.head = nn.Linear(in_features=embedding_dim, out_features=1)

        # Initialize
        # nn.init.uniform_(self.emb.weight, -1, 1)
        # nn.init.kaiming_normal_(self.neck.weight, mode='fan_in')
        # nn.init.kaiming_normal_(self.head.weight, mode='fan_in')

    def get_team_score(self, team):
        # Extract embeddings for each team member
        emb = self.emb(team)  # bs x team_dim -> bs x team_dim x  embedding_dim
        # Sum up team embeddings (all members contributed equally)
        emb = torch.sum(emb, 1)  # bs x team_dim x embedding_dim -> bs x  embedding_dim
        # Get score
        # neck = self.neck(emb)
        # return self.head(emb)
        return emb

    def get_loss(self, score_1, score_2, result):
        """
        Here we want to minimize the ranking error: team with higher score should win by default
        Consider the following cases:
        1. score_1 > score_2, result == 1: positive input, near-zero loss
        2. score_1 > score_2, result == -1: negative input, loss >> 0
        3. result == 0, force scores to be closer TODO: should loss be zero somehow?
        :param score_1: first team score
        :param score_2: second team score
        :param result: 1 if first team wins, -1 if second team wins, 0 if it's a tie
        :return: log sigmoid loss (details https://pytorch.org/docs/stable/nn.html#torch.nn.LogSigmoid)
        """
        # map scores between 0 and 1 (make them positive)
        # score_1 = nn.functional.sigmoid(score_1)
        # score_2 = nn.functional.sigmoid(score_2)

        score = torch.abs(torch.sigmoid(score_1 - score_2) * 2 - 1 - result)

        # result = (result + 1) / 2
        # score = result * score_1 + (1 - result) * score_2

        # score = -torch.log(score)
        return torch.mean(score)

        # score = (score_1 - score_2) * result
        # return torch.mean(-nn.functional.logsigmoid(score))

    def forward(self, team_1, team_2, result):
        # Get team scores
        score_1 = self.get_team_score(team_1)
        score_2 = self.get_team_score(team_2)

        # Clamp the score for better training (from word2vec -- do we need it?)
        # score_1 = torch.clamp(score_1, max=10, min=-10)
        # score_2 = torch.clamp(score_2, max=10, min=-10)

        # Get loss
        return self.get_loss(score_1, score_2, result)

