import pytest
import torch

from pretrain.optimizers.stable_adamw import StableAdamW


@pytest.mark.parametrize("weight_decay,decouple_lr", [(0.0, False), (0.01, False), (0.01, True)])
def test_stable_adamw_preserves_the_per_parameter_update(weight_decay, decouple_lr):
    generator = torch.Generator().manual_seed(241)
    initial = [torch.randn(shape, generator=generator) for shape in ((3, 4), (4,), (2, 3))]
    parameters = [torch.nn.Parameter(value.clone()) for value in initial]
    expected = [value.clone() for value in initial]
    averages, squares = [[torch.zeros_like(value) for value in initial] for _ in range(2)]
    lr, max_lr, eps = 2.4e-4, 1e-3, 1e-9
    beta1, beta2 = 0.9, 0.95
    optimizer = StableAdamW(
        parameters,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
        decouple_lr=decouple_lr,
        max_lr=max_lr,
    )
    for step in range(1, 5):
        gradients = [torch.randn(value.shape, generator=generator) for value in initial]
        if step == 2:
            gradients[0].zero_()
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient.clone()
        optimizer.step()
        first = 1 - (beta1**step - beta1) / (beta1**step - 1)
        second = (beta2**step - beta2) / (beta2**step - 1)
        for index, (parameter, gradient) in enumerate(zip(expected, gradients)):
            averages[index].lerp_(gradient, weight=first)
            squares[index].mul_(second).addcmul_(gradient, gradient, value=1 - second)
            stabilized = gradient.square() / torch.maximum(squares[index], torch.tensor(eps**2))
            rate = -lr / max(1, stabilized.mean().sqrt().item())
            if weight_decay:
                decay = rate / max_lr if decouple_lr else rate
                parameter.mul_(1 + decay * weight_decay)
            parameter.addcdiv_(averages[index], squares[index].sqrt().add_(eps), value=rate)
        for actual, wanted in zip(parameters, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for index, parameter in enumerate(parameters):
            torch.testing.assert_close(
                optimizer.state[parameter]["exp_avg"], averages[index], rtol=0, atol=0
            )
            torch.testing.assert_close(
                optimizer.state[parameter]["exp_avg_sq"], squares[index], rtol=0, atol=0
            )
