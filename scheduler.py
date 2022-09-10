from torch.optim.lr_scheduler import _LRScheduler
class Scheduler(_LRScheduler):
    def __init__(self, 
                 optimizer,
                 config, 
                 last_epoch: int=-1,
                 verbose: bool=False) -> None:

        self.dim_embed = config.EMBED_DIM
        self.warmup_steps = config.WARMUP_STEPS
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> float:
        lr = Scheduler.calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

    @staticmethod
    def calc_lr(step, dim_embed, warmup_steps):
        return dim_embed**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))