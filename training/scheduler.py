params_list = [[0.001] for _ in range(20)] +\
              [[0.0001] for _ in range(10)]


def count_epoch() -> int:
    return len(params_list)
