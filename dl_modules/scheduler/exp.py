import dl_modules.algorithm as algorithm
import dl_modules.warmup as warmup

# Minimum lerning rate, stop training if reached below
min_gen_lr = 0.00001


active = True
gen_lr = 0.0
times_decay = 0.0


def init(start_epoch: int, epoch_count: int, use_warmup: bool):
    global times_decay, gen_lr
    if use_warmup:
        epoch_count -= warmup.period
    times_decay = (min_gen_lr / algorithm.init_gen_lr) ** (1 / epoch_count)
    if start_epoch == 0:
        gen_lr = algorithm.init_gen_lr


def add_metrics(metrics: float) -> None:
    pass


def get_params() -> tuple:
    global gen_lr, times_decay
    gen_lr *= times_decay
    return gen_lr,


def discard():
    pass
