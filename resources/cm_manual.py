prepare = 'USAGE\n\npython prepare.py VIDEO_NAME [OPTIONS]\n\n' \
          '       --start={SECONDS, rSECONDS}\n' \
          '              video start time, applied to every episode (default 0.0),\n' \
          '              r prefix gives random frame from 0 to SECONDS\n\n' \
          '       -l=SECONDS, --length=SECONDS\n              processing time (default len(input_video))\n\n' \
          '       -s=SECONDS, --step=SECONDS\n              step between samples (default 1.0)\n\n' \
          '       -e=FIRST-LAST, --episodes=FIRST-LAST\n' \
          '              handle multiple episodes (format "VIDEO_NAME_00")\n\n' \
          '       -r=START_ID, --resume=START_ID\n              continue sample numeration from given number\n\n' \
          '       -d, --denoise\n              perform gentle denoising\n\n' \
          '       -h, --help\n              show manual\n\n' \
          'python prepare.py DATA_FOLDER {-c, --crop} [OPTIONS]\n\n' \
          '       -r=WIDTH:HEIGHT, --resolution=WIDTH:HEIGHT\n              crop size in pixels\n\n'\
          'python prepare.py DATA_FOLDER --extract [OPTIONS]\n\n' \
          '       -s=INT, --strength=INT\n              denoising strength (default 2)\n\n' \
          '       -w=SIZE, --window=SIZE\n              averaging area size (default 7)\n\n' \
          '       -k=SIZE, --kernel=SIZE\n              blur kernel size (default 5)\n\n' \
          '       --select=PATH_TO_LIST\n              separate patches with specific names\n\n' \
          'python prepare.py DATA_FOLDER {-g, --generate}\n\n' \
          '       generate lr images to create paired dataset\n'
