import abc

import torch
import torch.nn.functional as F

from catsample import sample_categorical


SUPPORTED_GRAPH_TYPES = ("absorb", "uniform")


def get_graph(config, device=None):
    graph_type = str(config.graph.type).strip().lower()
    if graph_type == "absorb":
        return Absorbing(int(config.tokens))
    if graph_type == "uniform":
        return Uniform(int(config.tokens))
    raise ValueError(
        f"Graph {config.graph.type!r} is not supported. "
        f"Use one of: {', '.join(SUPPORTED_GRAPH_TYPES)}."
    )


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):
    @property
    @abc.abstractmethod
    def dim(self):
        pass

    @property
    @abc.abstractmethod
    def absorb(self):
        pass

    @abc.abstractmethod
    def rate(self, i):
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        pass

    def sample_transition(self, i, sigma):
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        normalized_rate = self.transp_rate(i) * score
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(
            -1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True)
        )
        return normalized_rate

    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        pass


class Uniform(Graph):
    def __init__(self, dim):
        self._dim = int(dim)

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], -(self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        trans = (
            torch.ones(*i.shape, self.dim, device=i.device)
            * (1 - (-sigma).exp())
            / self.dim
        )
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans

    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        return torch.where(move_indices, torch.randint_like(i, self.dim), i)

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(
            dim=-1, keepdim=True
        ) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)
        ratio = 1 - self.dim / (esigm1 + self.dim)

        neg_term = score.mean(dim=-1) - torch.gather(
            score, -1, x[..., None]
        ).squeeze(-1) / self.dim
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term,
        )

        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim,
        )

        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(
            sexp, -1, x[..., None]
        ).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph):
    def __init__(self, dim):
        self._dim = int(dim)

    @property
    def dim(self):
        return self._dim + 1

    @property
    def absorb(self):
        return True

    def rate(self, i):
        mask = (self.dim - 1) * torch.ones_like(i)
        return F.one_hot(mask, num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        stay = (-sigma).exp()
        edge = stay * F.one_hot(i, num_classes=self.dim)
        edge[..., self.dim - 1] += 1 - stay.squeeze(-1)
        return edge

    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0,
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        return torch.where(move_indices, self.dim - 1, i)

    def staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, score, sigma, x, x0):
        rel_ind = x == self.dim - 1
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma), torch.exp(sigma) - 1)

        ratio = 1 / esigm1.expand_as(x)[rel_ind]
        other_ind = x0[rel_ind]

        neg_term = ratio * torch.gather(
            score[rel_ind], -1, other_ind[..., None]
        ).squeeze(-1)
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const
        return entropy
