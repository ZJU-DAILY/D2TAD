import torch


def get_model_fn(model, train=False):
    def model_fn(x, sigma, **kwargs):
        if train:
            model.train()
        else:
            model.eval()
        return model(x, sigma, **kwargs)

    return model_fn


def get_score_fn(model, train=False, sampling=False, extra_input=None):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.float32):
        def score_fn(x, sigma, **kwargs):
            sigma = sigma.reshape(-1)
            merged_kwargs = {}
            if extra_input is not None:
                merged_kwargs.update(
                    extra_input() if callable(extra_input) else dict(extra_input)
                )
            merged_kwargs.update(kwargs)
            score = model_fn(x, sigma, **merged_kwargs)
            if sampling:
                return score.exp()
            return score

    return score_fn
