from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, embedding_dim=1, take_best=6, loss='sigmoid'):
        super().__init__()
        assert loss in ('sigmoid', 'logsigmoid'), f'{loss} is invalid input'
        self.loss = loss
        self.take_best = take_best
        self.embedding_dim = embedding_dim

        # Embedding layer is sparse, it maps a player to a vector (initially zero)
        self.emb = nn.Embedding(num_embeddings=300000, embedding_dim=embedding_dim)
        torch.nn.init.zeros_(self.emb.weight)

        # If we want to constrain our weigths, we can apply clipper
        self.clipper = ZeroClipper()

        # hook to zero out gradient for idx 0
        def backward_hook(grad):
            out = grad.clone()
            out[0] = 0
            return out
        self.emb.weight.register_hook(backward_hook)

    def get_team_score(self, team):
        # Extract embeddings for each team member
        emb = self.emb(team)  # bs x team_dim -> bs x team_dim x embedding_dim
        # Take best players by embedding (sort in team_dim)
        emb, indices = torch.sort(emb, dim=1, descending=True)
        emb = emb[:, :self.take_best]
        # Sum up team embeddings (all members contributed equally)
        emb = torch.mean(emb, 1)  # bs x team_dim x embedding_dim -> bs x embedding_dim
        return emb

    def get_loss(self, score_1, score_2, result):
        """
        Here we want to minimize the ranking error: team with higher score should win by default
        Consider the following cases:
        1. score_1 > score_2, result == 1: positive input, near-zero loss (and vice verse)
        2. score_1 > score_2, result == -1: negative input, loss >> 0 (and vice versa)
        3. result == 0, force scores to be closer
        :param score_1: first team score
        :param score_2: second team score
        :param result: 1 if first team wins, -1 if second team wins, 0 if it's a tie
        :return: loss
        """

        # Ultimately, we optimize the delta between two scores with respect to the result
        delta = score_1 - score_2

        if self.loss == 'sigmoid':
            # Option 1: sigmoid loss
            # Problems: vanishing gradients
            score = torch.abs(torch.sigmoid(delta) * 2 - 1 - result)
        elif self.loss == 'logsigmoid':
            # Option 2: log sigmoid loss TODO test it
            # first term optimizes win/lose, second optimizes tie (effective if result == 0)
            score = -torch.nn.functional.logsigmoid(result * delta) + (1 - torch.abs(result)) * torch.abs(delta)
        else:
            raise ValueError('Invalid loss')

        return torch.mean(score)

    def forward(self, team_1, team_2, result):
        # Get team scores
        score_1 = self.get_team_score(team_1)
        score_2 = self.get_team_score(team_2)

        # Get loss
        return self.get_loss(score_1, score_2, result)


class ZeroClipper(object):
    """
    Clips model weights
    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933/3
    """
    def __init__(self, max_scale=False):
        self.max_scale = max_scale

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            m = torch.max(w)
            zero_indices = w == 0
            w.sub_(torch.min(w))

            if self.max_scale:
                # either scale everything between 0 and 1
                w.div_(torch.max(w))
            else:
                # or just prevent inflation (max w stays the same)
                w.div_(torch.max(w) / m)

            # Set zeroes back to zeroes after rescaling
            w[zero_indices] = 0

