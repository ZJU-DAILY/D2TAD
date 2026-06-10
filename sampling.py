import abc

import torch

from catsample import sample_categorical
from model import utils as mutils


_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    def _register(cls):
        local_name = cls.__name__ if name is None else name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered predictor: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma)
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        return self.graph.sample_rate(x, rev_rate)


@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma
        score = score_fn(x, curr_sigma)
        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t):
        sigma = self.noise(t)[0]
        score = score_fn(x, sigma)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        if self.graph.absorb:
            probs = probs[..., :-1]
        return sample_categorical(probs)


def get_sampling_fn(config, graph, noise, batch_dims, eps, device, model_extra=None):
    return get_pc_sampler(
        graph=graph,
        noise=noise,
        batch_dims=batch_dims,
        predictor=config.sampling.predictor,
        steps=config.sampling.steps,
        denoise=config.sampling.noise_removal,
        eps=eps,
        device=device,
        model_extra=model_extra,
    )


def get_pc_sampler(
    graph,
    noise,
    batch_dims,
    predictor,
    steps,
    denoise=True,
    eps=1e-5,
    device=torch.device("cpu"),
    proj_fun=lambda x: x,
    step_callback=None,
    model_extra=None,
):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        score_fn = mutils.get_score_fn(
            model, train=False, sampling=True, extra_input=model_extra
        )
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(score_fn, x, t, dt)
            if step_callback is not None:
                step_callback(i, x, t)

        if denoise:
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(score_fn, x, t)
            if step_callback is not None:
                step_callback(steps, x, t)

        return x

    return pc_sampler
