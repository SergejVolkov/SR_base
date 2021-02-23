import dl_modules.dataset as ds

usage = 'USAGE\n\npython main.py EPOCH_COUNT EXP_NAME [OPTIONS]\n\n' \
        '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              set CUDA device to use (default 0)\n\n' \
        '       -r, --resume\n              continue training from last checkpoint\n\n' \
        '       -s, --no_scheduler\n              do not use scheduler\n\n'\
        '       -w, --no_warmup\n              do not use warmup\n\n' \
        '       -p=MODEL_NAME, --pretrained=MODEL_NAME\n              load pretrained generator weights\n\n' \
        '       -b=COUNT, --batch=COUNT\n              set train batch size (default %d)\n\n'\
        '       -c=SIZE, --crop=SIZE\n              set train crop size (default %d)\n\n' \
        '       -t=SIZE, --train=SIZE\n              set train subset (default len(TRAIN_SET))\n\n' \
        '       -v=SIZE, --valid=SIZE\n              set valid subset (default len(VALID_SET))\n\n'\
        '       -h, --help\n              show manual\n\n' \
        'python main.py MODEL_NAME --predict [OPTIONS]\n\n'\
        '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              set CUDA device to use (default 0)\n\n' \
        '       -b=COUNT, --batch=COUNT\n              set predict batch size (default %d)\n\n'\
        'python main.py MODEL_NAME {-i, --inference} [OPTIONS]\n\n' \
        '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              set CUDA device to use (default 0)\n\n' \
        '       -s=SECONDS, --start=SECONDS\n              set predict video processing start time (default 0)\n\n'\
        '       -l=SECONDS, --length=SECONDS\n              set processing length time (default len(input_video))\n\n'\
        % (ds.train_batch_size, ds.crop_size, ds.valid_batch_size)
