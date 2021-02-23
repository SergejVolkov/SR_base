import dl_modules.scheduler as scheduler
import dl_modules.dataset as ds


# Number of warmup epochs
epoch_count = 4

# Warmup state
active = True

# Dataset size
total_samples = 0

# Current learning rate
gen_lr = 0
dis_lr = 0


def init():
    global total_samples
    total_samples = len(ds.train_loader)


def get_params(epoch_idx: int, sample_id: int) -> tuple:
    global active, gen_lr, dis_lr, epoch_count
    if epoch_idx == epoch_count:
        active = False
        gen_lr = scheduler.gen_lr
        dis_lr = scheduler.dis_lr
    else:
        coeff = (epoch_idx + (sample_id + 1) / total_samples) / epoch_count
        gen_lr = scheduler.gen_lr * coeff
        dis_lr = scheduler.dis_lr * coeff
    return gen_lr, dis_lr
