import dl_modules.algorithm as algotithm

# Metric type, 'loss' or 'acc'
metric_type = 'loss'

# Initial learning rate
gen_lr = algotithm.init_gen_lr
dis_lr = algotithm.init_dis_lr

# Divide lr by this number if metrics have platoed
power = 10.0

# Minimum lerning rate, stop training if reached below
min_gen_lr = 0.000001

# How many epoch to consider in diff computation.
# At least this number of epochs will be run with constant lr.
# Works better with even number
last_n_epoch = 16

# If diff is less than this, decrease learning rate
threshold = 0.001


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
    global gen_lr, dis_lr, epoch_counter, active, metric_type
    sign = 1
    if metric_type == 'loss':
        sign = -1
    if (epoch_counter >= last_n_epoch and
            sign * compute_diff(history, last_n_epoch) < threshold):
        epoch_counter = 0
        gen_lr /= power
        dis_lr /= power
        print('Learning rate descreased!')
    if gen_lr < min_gen_lr:
        active = False
    return gen_lr, dis_lr
