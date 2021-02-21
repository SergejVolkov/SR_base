import scripts.scheduler as scheduler
import scripts.dataset as ds


# Number of warmup epochs
epoch_count = 4

# Warmup state
active = True

# Dataset size
total_samples = 0

# Current learning rate
lr = 0


def init():
    global total_samples
    total_samples = len(ds.train_loader)


def get_params(epoch_idx: int, sample_id: int) -> tuple:
    global active, lr
    if epoch_idx == epoch_count:
        active = False
        lr = scheduler.lr
    else:
        lr = scheduler.lr * (epoch_idx +
                             (sample_id + 1) / total_samples) / epoch_count
    return lr,
