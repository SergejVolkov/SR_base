import sys
import torch
import os.path
import log_utils.log_tb as log
import resources.dl_manual as man
import dl_modules.dataset as ds
import dl_modules.algorithm as algorithm
import dl_modules.scheduler.exp as scheduler
import dl_modules.warmup as warmup
from dl_modules.train import train
from dl_modules.valid import valid, simple_eval
# from dl_modules.valid import get_static_images
from cm_modules.predict import predict
from cm_modules.inference import inference
from models.RDN import RDN
from models.RevDiscr import RevDiscr
from models.Algo import Bicubic


def train_start_log(device: torch.device):
    # Evaluate naive solution for future comparison
    naive = Bicubic()
    naive.to(device)
    valid_psnr, valid_ssim, valid_lpips = \
        simple_eval(naive, device, bars=True, title="Valid Bicubic")
    print('Bicubic: PSNR: %.2f, SSIM: %.4f, LPIPS: %.4f' %
          (valid_psnr, valid_ssim, valid_lpips))
    # log.add(epoch_idx=0, constants=(valid_psnr, valid_ssim, valid_lpips))

    # Add static images to log
    # log.add(epoch_idx=0, images=tuple(get_static_images()), im_start=6)


def start_train():
    if len(sys.argv) < 4:
        print('Wrong number of params!\nTry "python main.py --help" for usage info')
        return

    epoch_count = int(sys.argv[2])
    exp_name = sys.argv[3]
    pretrained = dis_weights = None
    start_epoch = 0
    best_result = float('inf')
    use_scheduler = use_warmup = False
    use_bars = False
    resume = False
    cuda_id = 0
    for arg in sys.argv[4:]:
        if arg == '-r' or arg == '--resume':
            PATH = ds.SAVE_DIR + 'weights/checkpoint'
            if not os.path.isfile(PATH):
                print('Cannot resume training, no saved checkpoint found!')
                exit(0)
            resume = True
        elif arg == '-s' or arg == '--scheduler':
            use_scheduler = True
        elif arg == '--bars':
            use_bars = True
        elif arg.startswith('-w=') or arg.startswith('--warmup='):
            use_warmup = True
            warmup.period = int(arg[arg.index('=') + 1:])
        elif arg.startswith('--gen_lr='):
            algorithm.init_gen_lr = float(arg[arg.index('=') + 1:])
        elif arg.startswith('--min_gen_lr='):
            scheduler.min_gen_lr = float(arg[arg.index('=') + 1:])
        elif arg.startswith('--dis_lr='):
            algorithm.dis_lr = float(arg[arg.index('=') + 1:])
        elif arg.startswith('-p=') or arg.startswith('--pretrained='):
            pretrained = arg[arg.index('=') + 1:]
        elif arg.startswith('-d=') or arg.startswith('--dis_weights='):
            dis_weights = arg[arg.index('=') + 1:]
        elif arg.startswith('-b=') or arg.startswith('--batch='):
            ds.train_batch_size = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-c=') or arg.startswith('--crop='):
            ds.crop_size = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-v=') or arg.startswith('--valid='):
            ds.valid_set_size = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-t=') or arg.startswith('--train='):
            ds.train_set_size = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-g=') or arg.startswith('--gpu='):
            cuda_id = int(arg[arg.index('=') + 1:])
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    # Try to use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device, 'hardware:%d' % cuda_id)

    # Init datasets, metrics and logger
    ds.init_data()
    log.init(exp_name)
    if not resume:
        train_start_log(device)

    # Create an instance of the model
    generator = RDN(ds.scale, 3, 64, 64, 16, 8)
    generator.to(device)
    discriminator = RevDiscr(6, 64)
    discriminator.to(device)

    # Resume from the last checkpoint or load pretrained weights
    if resume:
        PATH = ds.SAVE_DIR + 'weights/checkpoint'
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_acc']
        scheduler.gen_lr = checkpoint['lr']
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        algorithm.gen_opt_state_dict = checkpoint['gen_optimizer']
        algorithm.dis_opt_state_dict = checkpoint['dis_optimizer']
        use_warmup = False
    else:
        if pretrained is not None:
            PATH = ds.SAVE_DIR + 'weights/' + pretrained + '.pth'
            generator.load_state_dict(torch.load(PATH))
        if dis_weights is not None:
            PATH = ds.SAVE_DIR + 'weights/' + dis_weights + '.pth'
            discriminator.load_state_dict(torch.load(PATH))

    if start_epoch == epoch_count:
        print('Cannot resume training, already reached last epoch!')
        return

    # Train model
    train(generator, discriminator, device, epoch_count=epoch_count, start_epoch=start_epoch,
          use_scheduler=use_scheduler, use_warmup=use_warmup, best_accuracy=best_result, bars=use_bars)

    # Test model on all valid data
    if ds.valid_set_size != 0:
        ds.valid_set_size = 0
        ds.init_data()
        generator.eval()
        discriminator.eval()
        valid_psnr, valid_ssim, valid_lpips, valid_gen_loss,\
            valid_dis_loss, _ = valid(generator, discriminator, device,
                                      save_images=False, bars=True, title="Valid Full")
        print('Full valid: GEN loss: %.3f, DIS loss: %.3f\n'
              'Full valid: PSNR: %.2f, SSIM: %.4f, LPIPS: %.4f' %
              (valid_gen_loss, valid_dis_loss,
               valid_psnr, valid_ssim, valid_lpips))


def start_predict():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python main.py --help" for usage info')
        return

    pretrained = sys.argv[2]

    cuda_id = 0
    for arg in sys.argv[3:]:
        if arg.startswith('-g=') or arg.startswith('--gpu='):
            cuda_id = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-b=') or arg.startswith('--batch='):
            ds.valid_batch_size = int(arg[arg.index('=') + 1:])
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    # Try to use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device, 'hardware:%d' % cuda_id)

    # Create an instance of the model
    if pretrained == 'algo':
        generator = Bicubic()
        generator.to(device)
    else:
        generator = RDN(ds.scale, 3, 64, 64, 16, 8)
        generator.to(device)
        PATH = ds.SAVE_DIR + 'weights/' + pretrained + '.pth'
        generator.load_state_dict(torch.load(PATH))

    # Inference model on images in 'predict' folder
    predict(generator, device)


def start_inference():
    if len(sys.argv) < 4:
        print('Wrong number of params!\nTry "python main.py --help" for usage info')
        return

    pretrained = sys.argv[2]
    name = sys.argv[3]
    length = start = 0
    cuda_id = 0
    for arg in sys.argv[4:]:
        if arg.startswith('-g=') or arg.startswith('--gpu='):
            cuda_id = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-s=') or arg.startswith('--start='):
            start = int(arg[arg.index('=') + 1:])
        elif arg.startswith('-l=') or arg.startswith('--length='):
            length = int(arg[arg.index('=') + 1:])
        else:
            print('Unexpected argument "' + arg + '"!')
            return

    # Try to use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device, 'hardware:%d' % cuda_id)

    # Create an instance of the model
    if pretrained == 'algo':
        generator = Bicubic()
        generator.to(device)
    else:
        generator = RDN(ds.scale, 3, 64, 64, 16, 8)
        generator.to(device)
        PATH = ds.SAVE_DIR + 'weights/' + pretrained + '.pth'
        generator.load_state_dict(torch.load(PATH))

    # Process video in 'video' folder
    inference(name, generator, device, length, start)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.train)
        else:
            start_train()
    elif sys.argv[1] == 'predict':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.predict)
        else:
            start_predict()
    elif sys.argv[1] == 'inference':
        if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
            print(man.inference)
        else:
            start_inference()
    elif sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
        print(man.common)
    else:
        print('No jobs to do.\nTry "python main.py --help" for usage info')
