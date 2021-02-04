import scripts.scheduler as scheduler
import scripts.dataset as ds


# Number of warmup epochs
epoch_count = 2

# Warmup state
active = True

# Dataset size
total_samples = len(ds.train_loader)


def get_params(epoch_idx: int, sample_id: int) -> tuple:
    global active
    if epoch_idx == epoch_count:
        active = False
        return scheduler.lr
    return (scheduler.lr * (epoch_idx +
                                  (sample_id + 1) / total_samples) / epoch_count, )
