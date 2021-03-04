import dl_modules.algorithm as algorithm
import dl_modules.dataset as ds


# Number of warmup epochs
period = 4

# Warmup state
active = True

# Dataset size
total_samples = 0

# Current learning rate
gen_lr = 0


def init():
    global total_samples
    total_samples = len(ds.train_loader)


def get_params(epoch_idx: int, sample_id: int) -> tuple:
    global active, gen_lr, period
    if epoch_idx == period:
        active = False
        gen_lr = algorithm.init_gen_lr
    else:
        coeff = (epoch_idx + (sample_id + 1) / total_samples) / period
        gen_lr = algorithm.init_gen_lr * coeff
    return gen_lr,
