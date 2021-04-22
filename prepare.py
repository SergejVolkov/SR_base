import io
import sys
import resources.cm_manual as man
import dl_modules.dataset as ds
from cm_modules.slice import slice_data
from cm_modules.crop import crop
from cm_modules.extract import extract
from cm_modules.generate_lr import generate
from cm_modules.enhance import enhance_images
from cm_modules.draw import draw_data
from cm_modules.draw import extract_palette
from cm_modules.draw import random_palette


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
    kernel = 0
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
    if len(sys.argv) < 2:
        print('Wrong number of params!\nTry "python prepare.py --help" for usage info')
        return

    folder = None
    if len(sys.argv) == 3:
        folder = sys.argv[2]
    elif len(sys.argv) > 3:
        print('Unexpected argument "' + sys.argv[3] + '"!')
        return

    # Process images in folder
    generate(folder)


def start_enhance():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python prepare.py --help" for usage info')
        return

    folder = sys.argv[2]
    strength = 4
    window_size = 5
    contrast = 5
    kernel = 5
    for arg in sys.argv[3:]:
        if arg.startswith('-s=') or arg.startswith('--strength='):
            strength = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-w=') or arg.startswith('--window='):
            window_size = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-c=') or arg.startswith('--contrast='):
            contrast = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-k=') or arg.startswith('--kernel='):
            kernel = int(arg[arg.index('=') + 1:])
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    # Process images in folder
    enhance_images(folder, strength, window_size, contrast, kernel)


def start_draw():
    if len(sys.argv) < 4:
        print('Wrong number of params!\nTry "python prepare.py --help" for usage info')
        return

    folder = sys.argv[2]
    count = int(sys.argv[3])
    width = height = sample_id = 0
    line_count = 100
    kernel = 11
    source = palette_folder = None
    for arg in sys.argv[4:]:
        if arg.startswith('--resume='):
            sample_id = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-l=') or arg.startswith('--lines='):
            line_count = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-p=') or arg.startswith('--palette='):
            palette_folder = arg[arg.index('=') + 1:]
        elif arg.startswith('-k=') or arg.startswith('--kernel='):
            kernel = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-r=') or arg.startswith('--resolution='):
            dash_ind = arg.index(':')
            width = int(arg[arg.index('=') + 1: dash_ind])
            height = int(arg[dash_ind + 1:])
        elif arg.startswith('-s=') or arg.startswith('--source='):
            source = arg[arg.index('=') + 1:]
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    if source is None:
        if palette_folder is not None:
            palette = extract_palette(palette_folder, kernel)
        else:
            palette = random_palette(count)
    else:
        palette = None

    # Process images in folder
    draw_data(folder, palette, (width, height), count, line_count, sample_id, source)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('No jobs to do.\nTry "python main.py --help" for usage info')
    elif sys.argv[1] == 'slice':
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
    elif sys.argv[1] == 'enhance':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.enhance)
        else:
            start_enhance()
    elif sys.argv[1] == 'draw':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.draw)
        else:
            start_draw()
    elif sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
        print(man.common)
    else:
        print('Wrong job!\nTry "python main.py --help" for usage info')
