import math
import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

class WarmupLinearSchedule(LambdaLR): 
    """ 
    Linear warmup and then linear decay on an epoch basis.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_epochs, t_total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.t_total_epochs = t_total_epochs
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return float(epoch) / float(max(1, self.warmup_epochs))
        return max(0.0, float(self.t_total_epochs - epoch) / float(max(1.0, self.t_total_epochs - self.warmup_epochs)))


class WarmupCosineSchedule(LambdaLR):
    """
    Linear warmup and then cosine decay on an epoch basis.
    Linearly increases learning rate from 0 to 1 over `warmup_epochs` epochs.
    Decreases learning rate from 1. to 0. over remaining `t_total_epochs - warmup_epochs` epochs following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_epochs, t_total_epochs, cycles=0.5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.t_total_epochs = t_total_epochs
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, epoch):
        if epoch < self.warmup_epochs:
            return float(epoch) / float(max(1.0, self.warmup_epochs))
        # Progress after warmup
        progress = float(epoch - self.warmup_epochs) / float(max(1, self.t_total_epochs - self.warmup_epochs))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, T_max, cycles=0.5, eta_min=0, last_epoch=-1):
        """
        Combines warmup and cosine annealing learning rate scheduling.
        
        optimizer: The optimizer to which the learning rate scheduler is attached.
        warmup_epochs: The number of epochs for linear warmup.
        T_max: The total number of epochs for cosine annealing after warmup.
        cycles: The number of cosine cycles to complete after warmup.
        eta_min: The minimum learning rate at the end of the annealing.
        last_epoch: The index of the last epoch (default=-1).
        """
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.cycles = cycles
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Current epoch
        epoch = self.last_epoch

        # Warmup phase
        if epoch < self.warmup_epochs:
            lr = [self.eta_min + base_lr * (epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase after warmup
            progress = (epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * self.cycles * 2 * progress)))
            lr = [base_lr * (cosine_factor) + self.eta_min for base_lr in self.base_lrs]

        return lr