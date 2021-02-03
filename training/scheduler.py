params_list = [[0.0005] for _ in range(20)] +\
              [[0.00001] for _ in range(10)]


def count_epoch() -> int:
    return len(params_list)
