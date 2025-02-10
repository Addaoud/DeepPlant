from torch import optim
import math


class CosineScheduler:
    """https://d2l.ai/chapter_optimization/lr-scheduler.html"""

    def __init__(
        self,
        max_epochs: int,
        max_lr=0.01,
        final_lr=0,
        warmup_steps=0,
        warmup_begin_lr=0,
    ):
        self.max_lr = max_lr
        self.max_epochs = max_epochs
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_epochs - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (
            (self.max_lr - self.warmup_begin_lr)
            * float(epoch)
            / float(self.warmup_steps)
        )
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        elif epoch <= self.max_epochs:
            return (
                self.final_lr
                + (self.max_lr - self.final_lr)
                * (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps))
                / 2
            )
        else:
            return self.final_lr


class ScheduledOptim:
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, config):
        self.config = config
        self.lr_schedule = CosineScheduler(
            config.max_epochs,
            max_lr=config.max_lr,
            final_lr=config.final_lr,
            warmup_steps=config.warmup_steps,
            warmup_begin_lr=config.warmup_begin_lr,
        )
        self.lr = self.lr_schedule(0)

    def __call__(self, model_parameters):
        if self.config.optimizer_type.upper() == "ADAMW":
            self.optimizer = optim.AdamW(
                model_parameters,
                lr=self.lr,
                betas=(0.9, 0.99),
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type.upper() == "SGD":
            self.optimizer = optim.SGD(
                model_parameters,
                lr=self.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(model_parameters, lr=self.lr)

    def step(self):
        """Step within the inner optimizer"""
        self.optimizer.step()

    def update_lr(self, epoch):
        """Update the learning rate"""
        self.lr = self.lr_schedule(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def zero_grad(self):
        """Zero out the gradients within the inner optimizer"""
        self.optimizer.zero_grad()
