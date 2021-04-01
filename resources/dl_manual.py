import dl_modules.dataset as ds
import dl_modules.algorithm as algorithm
import dl_modules.scheduler.exp as scheduler

common = \
    'USAGE\n\npython main.py {train, predict, inference} [OPTIONS]\n\n' \
    '       train\n              train model\n\n' \
    '       predict\n              process images\n\n' \
    '       inference\n              process video\n\n' \
    '       unpack\n              extract weights from checkpoint\n\n' \
    '       -h, --help\n              show manual\n'

train = \
    '\npython main.py train EPOCH_COUNT EXP_NAME [OPTIONS]\n\n' \
    '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              CUDA device to use (default 0)\n\n' \
    '       -r, --resume\n              continue training from last checkpoint\n\n' \
    '       -s, --scheduler\n              use scheduler\n\n' \
    '       --gen_lr=LR\n              initial generator learning rate (default %g)\n\n' \
    '       --min_gen_lr=LR\n              minimum generator learning rate at the end of training (default %g)\n\n' \
    '       --dis_lr=LR\n              discriminator learning rate (default %g)\n\n' \
    '       -w=EPOCH_COUNT, --warmup=EPOCH_COUNT\n              use warmup during specified period\n\n' \
    '       -p=MODEL_NAME, --pretrained=MODEL_NAME\n              load pretrained generator weights\n\n' \
    '       -d=MODEL_NAME, --dis_weights=MODEL_NAME\n              load pretrained discriminator weights\n\n' \
    '       -b=COUNT, --batch=COUNT\n              train batch size (default %d)\n\n' \
    '       -c=SIZE, --crop=SIZE\n              train crop size (default %d)\n\n' \
    '       -t=SIZE, --train=SIZE\n              train subset (default len(TRAIN_SET))\n\n' \
    '       -v=SIZE, --valid=SIZE\n              valid subset (default len(VALID_SET))\n\n' \
    '       --bars\n              show progressbars\n' \
    % (algorithm.init_gen_lr, scheduler.min_gen_lr, algorithm.dis_lr,
       ds.train_batch_size, ds.crop_size)

predict = \
    '\npython main.py predict MODEL_NAME [OPTIONS]\n\n' \
    '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              CUDA device to use (default 0)\n\n' \
    '       -b=COUNT, --batch=COUNT\n              predict batch size (default %d)\n\n' \
    '       -c, --cut\n              cut image and use model separately on each piece to reduce cuda memory\n\n' \
    '       -e, --enhance\n              process super-resolved images with some cool algo to make it look better\n' \
    % ds.valid_batch_size

inference = \
    '\npython main.py inference MODEL_NAME VIDEO_NAME [OPTIONS]\n\n' \
    '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              CUDA device to use (default 0)\n\n' \
    '       -s=SECONDS, --start=SECONDS\n              predict video start time (default 0)\n\n' \
    '       -l=SECONDS, --length=SECONDS\n              processing time (default len(input_video))\n\n' \
    '       -c, --cut\n              cut image and use model separately on each piece to reduce cuda memory\n\n' \
    '       -e, --enhance\n              process super-resolved video with some cool algo to make it look better\n'

unpack = \
    '\npython main.py unpack [CHECKPOINT_NAME]\n\n' \
    '       CHECKPOINT_NAME\n              default "checkpoint"\n'
