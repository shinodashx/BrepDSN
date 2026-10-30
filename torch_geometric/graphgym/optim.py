from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

from torch.nn import Parameter
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, \
    StepLR, ExponentialLR, LambdaLR, SequentialLR, CosineAnnealingWarmRestarts

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import from_config


@dataclass
class OptimizerConfig:
    optimizer: str = 'adam'  # ['sgd', 'adam']
    base_lr: float = 0.01
    weight_decay: float = 5e-4
    momentum: float = 0.9  # 'sgd' policy


@register.register_optimizer('adam')
def adam_optimizer(params: Iterator[Parameter], base_lr: float, weight_decay: float) -> Adam:
    return Adam(params, lr=base_lr, weight_decay=weight_decay)


@register.register_optimizer('sgd')
def sgd_optimizer(params: Iterator[Parameter], base_lr: float, momentum: float, weight_decay: float) -> SGD:
    return SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)


def create_optimizer(params: Iterator[Parameter], cfg: Any) -> Any:
    r"""Creates a config-driven optimizer."""
    params = filter(lambda p: p.requires_grad, params)
    func = register.optimizer_dict.get(cfg.optimizer, None)
    if func is not None:
        return from_config(func)(params, cfg=cfg)
    raise ValueError(f"Optimizer '{cfg.optimizer}' not supported")


@dataclass
class SchedulerConfig:
    scheduler: Optional[str] = 'cos'  # [None, 'steps', 'cos']
    steps: List[int] = field(default_factory=[30, 60, 90])  # 'steps' policy
    lr_decay: float = 0.1  # 'steps' policy
    max_epoch: int = 200


@register.register_scheduler(None)
@register.register_scheduler('none')
def none_scheduler(optimizer: Optimizer, max_epoch: int) -> StepLR:
    return StepLR(optimizer, step_size=max_epoch + 1)


@register.register_scheduler('step')
def step_scheduler(optimizer: Optimizer, steps: List[int],
                   lr_decay: float) -> MultiStepLR:
    return MultiStepLR(optimizer, milestones=steps, gamma=lr_decay)


@register.register_scheduler('cos')
def cos_scheduler(optimizer: Optimizer, max_epoch: int) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=max_epoch)


# 以下是新加入的一些scheduler；
@register.register_scheduler('exp')
def exp_scheduler(optimizer: Optimizer, gamma):
    return ExponentialLR(optimizer, gamma=gamma)


@register.register_scheduler('cos_constant')
def chained_scheduler_0(optimizer, milestones, eta_min, max_epoch, const_factor):
    # 使用cosine annealing scheduler + constant schedular
    scheduler0 = CosineAnnealingLR(optimizer, T_max=milestones, eta_min=eta_min)
    # Define the second scheduler: Constant
    scheduler1 = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1)
    scheduler = SequentialLR(optimizer, [scheduler0, scheduler1], milestones=[milestones])
    # sequentialLR在切换至另一个schedular时，会在内部执行scheduler.step(0)，这说明不同的schedular的epoch独立计数；
    return scheduler


@register.register_scheduler('cos_multi')
def cos_multi_scheduler(optimizer, T_0, T_mult, eta_min):
    # T_0表示第一次 learning rate从最大衰减到最小需要多少个step或epoch;
    # T_mult表示后续每一次衰减所需要的step或epoch以哪种 scale 增长;
    return CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=T_mult, eta_min=eta_min)


def create_scheduler(optimizer: Optimizer, cfg: Any) -> Any:
    r"""Creates a config-driven learning rate scheduler."""
    func = register.scheduler_dict.get(cfg.scheduler, None)
    if func is not None:
        return from_config(func)(optimizer, cfg=cfg)
    raise ValueError(f"Scheduler '{cfg.scheduler}' not supported")
