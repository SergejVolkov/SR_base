common = \
    'USAGE\n\npython prepare.py {slice, crop, extract, generate} [OPTIONS]\n\n' \
    '       slice\n              extract frames from mp4 videos to create dataset\n\n' \
    '       crop\n              crop cinematic/TV resolution letterbox if needed\n\n' \
    '       extract\n              extract noise patches from dataset\n\n' \
    '       generate\n              generate lr images to create paired dataset\n\n' \
    '       enhance\n              process images with some cool algo to make them look better\n\n' \
    '       -h, --help\n              show manual\n'

slice = \
    '\npython prepare.py slice VIDEO_NAME [OPTIONS]\n\n' \
    '       --start={SECONDS, rSECONDS}\n' \
    '              video start time, applied to every episode (default 0.0),\n' \
    '              r prefix gives random frame from 0 to SECONDS\n\n' \
    '       -l=SECONDS, --length=SECONDS\n              processing time (default len(input_video))\n\n' \
    '       -s=SECONDS, --step=SECONDS\n              step between samples (default 1.0)\n\n' \
    '       -e=FIRST-LAST, --episodes=FIRST-LAST\n' \
    '              handle multiple episodes (format "VIDEO_NAME_00")\n\n' \
    '       -r=START_ID, --resume=START_ID\n              continue sample numeration from given number\n\n' \
    '       -d, --denoise\n              perform gentle denoising\n' \

crop = \
    '\npython prepare.py crop DATA_FOLDER [OPTIONS]\n\n' \
    '       -r=WIDTH:HEIGHT, --resolution=WIDTH:HEIGHT\n              crop size in pixels\n'

extract = \
    '\npython prepare.py extract DATA_FOLDER [OPTIONS]\n\n' \
    '       -s=INT, --strength=INT\n              denoising strength (default 2)\n\n' \
    '       -w=SIZE, --window=SIZE\n              averaging area size (default 7)\n\n' \
    '       -k=SIZE, --kernel=SIZE\n              blur kernel size (default 0)\n\n' \
    '       --select=PATH_TO_LIST\n              separate patches with specific names\n'

generate = \
    '\npython prepare.py generate DATA_FOLDER\n'

enhance = \
    '\npython prepare.py enhance DATA_FOLDER [OPTIONS]\n\n' \
    '       -s=INT, --strength=INT\n              denoising strength (default 4)\n\n' \
    '       -w=SIZE, --window=SIZE\n              averaging area size (default 5)\n\n' \
    '       -c=INT, --contrast=INT\n              auto contrast level (default 5)\n\n' \
    '       -k=SIZE, --kernel=SIZE\n              gentle noise extraction level (default 5)\n'
