import io
import sys
import resources.cm_manual as man
import dl_modules.dataset as ds
from cm_modules.slice import slice_data
from cm_modules.crop import crop
from cm_modules.extract import extract
from cm_modules.generate_lr import generate


def start_slice():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python prepare.py --help" for usage info')
        return

    name = sys.argv[2]
    denoise = episodes = random = False
    ep_start = ep_end = 0
    length = start = 0.0
    sample_id = 0
    step = 1.0
    for arg in sys.argv[3:]:
        if arg.startswith('--start=r'):
            random = True
            start = float(arg[arg.index('=r') + 2:])
        elif arg.startswith('--start='):
            start = float(arg[arg.index('=') + 1:])
        elif arg == '-d' or arg == '--denoise=':
            denoise = True
        elif arg.startswith('-s=') or arg.startswith('--step='):
            step = float(arg[arg.index('=') + 1:])
        elif arg.startswith('-l=') or arg.startswith('--length='):
            length = float(arg[arg.index('=') + 1:])
        elif arg.startswith('-e=') or arg.startswith('--episodes='):
            episodes = True
            dash_ind = len(arg) - arg[::-1].index('-') - 1
            ep_start = int(arg[arg.index('=') + 1: dash_ind])
            ep_end = int(arg[dash_ind + 1:])
        elif arg.startswith('-r=') or arg.startswith('--resume='):
            sample_id = int(arg[arg.index('=') + 1:])
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    # Process video in 'video' folder
    slice_data(name, step, length, start, random, episodes, ep_start, ep_end, sample_id, denoise)


def start_crop():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python prepare.py --help" for usage info')
        return

    folder = sys.argv[2]
    width = height = 0
    for arg in sys.argv[3:]:
        if arg.startswith('-r=') or arg.startswith('--resolution='):
            dash_ind = arg.index(':')
            width = int(arg[arg.index('=') + 1: dash_ind])
            height = int(arg[dash_ind + 1:])
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    # Process images in folder
    crop(folder, width, height)


def start_extract():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python prepare.py --help" for usage info')
        return

    folder = sys.argv[2]
    strength = 2
    window_size = 7
    kernel = 5
    select_file = None
    for arg in sys.argv[3:]:
        if arg.startswith('--select='):
            select_file = arg[arg.index('=') + 1:]
        elif arg.startswith('-s=') or arg.startswith('--strength='):
            strength = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-w=') or arg.startswith('--window='):
            window_size = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-k=') or arg.startswith('--kernel='):
            kernel = int(arg[arg.index('=') + 1:])
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    select = None
    if select_file is not None:
        f = io.open(ds.SAVE_DIR + 'data/' + select_file)
        select = [line[:-1] for line in f.readlines()]

    # Process images in folder
    extract(folder, strength, window_size, kernel, select)


def start_generate():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python prepare.py --help" for usage info')
        return

    folder = sys.argv[2]
    if len(sys.argv) > 3:
        print('Unexpected argument "' + sys.argv[3] + '"!')
        return

    # Process images in folder
    generate(folder)


if __name__ == "__main__":
    if sys.argv[1] == 'slice':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.slice)
        else:
            start_slice()
    elif sys.argv[1] == 'crop':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.crop)
        else:
            start_crop()
    elif sys.argv[1] == 'extract':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.extract)
        else:
            start_extract()
    elif sys.argv[1] == 'generate':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.generate)
        else:
            start_generate()
    elif sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
        print(man.common)
    else:
        print('No jobs to do.\nTry "python main.py --help" for usage info')
