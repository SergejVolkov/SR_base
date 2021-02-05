# Initial leraning rate
lr = 0.001

# Divide lr by this number if mertics have platoed
power = 10.0

# Minimum lerning rate, stop training if reached below
min_lr = 0.00001

# How many epoch to consider in diff computation.
# At least this number of epochs will be run with constant lr.
# Better if even number
last_n_epoch = 10

# If diff is less than this, decrease learning rate
threshold = 0.12


active = True
history = []
epoch_counter = 0


def compute_diff(metrics: list, window_size: int) -> float:
    avg1 = 0
    for i in range(window_size // 2, window_size):
        avg1 += metrics[-i - 1]
    avg1 /= window_size - window_size // 2
    avg2 = 0
    for i in range(window_size // 2):
        avg2 += metrics[-i - 1]
    avg2 /= window_size // 2
    # print('avg', (avg2 - avg1) * 2.0, avg1, avg2)
    return (avg2 - avg1) * 2.0


def add_metrics(metrics: float) -> None:
    global history, epoch_counter
    history.append(metrics)
    epoch_counter += 1


def get_params() -> tuple:
    global lr, epoch_counter, active
    if epoch_counter >= last_n_epoch \
            and compute_diff(history, last_n_epoch) < threshold:
        epoch_counter = 0
        lr /= power
    if lr < min_lr:
        active = False
    return lr,
