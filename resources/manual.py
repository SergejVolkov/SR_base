import dl_modules.dataset as ds
import dl_modules.algorithm as algorithm
import dl_modules.scheduler as scheduler

main = 'USAGE\n\npython main.py EPOCH_COUNT EXP_NAME [OPTIONS]\n\n' \
       '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              CUDA device to use (default 0)\n\n' \
       '       -r, --resume\n              continue training from last checkpoint\n\n' \
       '       -s, --scheduler\n              use scheduler\n\n' \
       '       -w, --warmup\n              use warmup\n\n' \
       '       --gen_lr=LR\n              initial generator learning rate (default %g)\n\n' \
       '       --min_gen_lr=LR\n              minimum generator learning rate at the end of training (default %g)\n\n' \
       '       --dis_lr=LR\n              discriminator learning rate (default %g)\n\n' \
       '       -p=MODEL_NAME, --pretrained=MODEL_NAME\n              load pretrained generator weights\n\n' \
       '       -b=COUNT, --batch=COUNT\n              train batch size (default %d)\n\n' \
       '       -c=SIZE, --crop=SIZE\n              train crop size (default %d)\n\n' \
       '       -t=SIZE, --train=SIZE\n              train subset (default len(TRAIN_SET))\n\n' \
       '       -v=SIZE, --valid=SIZE\n              valid subset (default len(VALID_SET))\n\n' \
       '       --bars\n              show progressbars\n\n' \
       '       -h, --help\n              show manual\n\n' \
       'python main.py MODEL_NAME --predict [OPTIONS]\n\n' \
       '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              CUDA device to use (default 0)\n\n' \
       '       -b=COUNT, --batch=COUNT\n              predict batch size (default %d)\n\n' \
       'python main.py MODEL_NAME VIDEO_NAME {-i, --inference} [OPTIONS]\n\n' \
       '       -g=CUDA_DEVICE_NUMBER, --gpu=CUDA_DEVICE_NUMBER\n              CUDA device to use (default 0)\n\n' \
       '       -s=SECONDS, --start=SECONDS\n              predict video start time (default 0)\n\n' \
       '       -l=SECONDS, --length=SECONDS\n              processing time (default len(input_video))\n' \
       % (algorithm.init_gen_lr, scheduler.min_gen_lr, algorithm.dis_lr,
          ds.train_batch_size, ds.crop_size, ds.valid_batch_size)

prepare = 'USAGE\n\npython prepare.py VIDEO_NAME [OPTIONS]\n\n' \
          '       --start={SECONDS, rSECONDS}\n' \
          '              video start time, applied to every episode (default 0.0),\n' \
          '              r prefix gives random frame from 0 to SECONDS\n\n' \
          '       -l=SECONDS, --length=SECONDS\n              processing time (default len(input_video))\n\n' \
          '       -s=SECONDS, --step=SECONDS\n              step between samples (default 1.0)\n\n' \
          '       -e=FIRST-LAST, --episodes=FIRST-LAST\n' \
          '              handle multiple episodes (format "VIDEO_NAME_00")\n\n' \
          '       -r=START_ID, --resume=START_ID\n              continue sample numeration from given number\n\n' \
          '       -h, --help\n              show manual\n\n' \
          'python prepare.py DATA_FOLDER {-c, --crop} [OPTIONS]\n\n' \
          '       -r=WIDTH:HEIGHT, --resolution=WIDTH:HEIGHT\n              crop size in pixels (default image size)\n'
