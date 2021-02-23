import sys
import torch
import os.path
import log_utils.log_tb as log
import dl_modules.dataset as ds
import dl_modules.algorithm as algorithm
import resources.manual as man
from models.RDN import RDN
from models.SimpleDiscr import ConvDiscr
# from models.Algo import Bicubic
from dl_modules.train import train
from dl_modules.valid import valid
from dl_modules.valid import get_static_images
from dl_modules.predict import predict
from dl_modules.inference import inference


def train_start_log():
    # Evaluate naive solution for future comparison
    # naive = Bicubic()
    # naive.to(device)
    # naive_acc, naive_loss, _ = valid(naive, device, save_images=False, title="Valid Bicubic")
    # print('Bicubic loss: %.3f, bicubic accuracy: %.3f' % (naive_loss, naive_acc))
    # log.add(epoch_idx=0, constants=(naive_acc, naive_loss))

    # Add static images to log
    log.add(epoch_idx=0, images=tuple(get_static_images()), im_start=6)


def start_train():
    if len(sys.argv) < 4:
        print('Wrong number of params!\nTry "python main.py --help" for usage info')
        return

    epoch_count = int(sys.argv[1])
    exp_name = sys.argv[2]
    pretrained = None
    start_epoch = 0
    best_result = float('inf')
    use_scheduler = use_warmup = True
    resume = False
    cuda_id = 0
    for arg in sys.argv[3:]:
        if arg == '-r' or arg == '--resume':
            PATH = ds.SAVE_DIR + 'model_instances/checkpoint'
            if not os.path.isfile(PATH):
                print('Cannot resume training, no saved checkpoint found!')
                exit(0)
            resume = True
        elif arg == '-s' or arg == '--no_scheduler':
            use_scheduler = False
        elif arg == '-w' or arg == '--no_warmup':
            use_warmup = False
        elif arg.startswith('-p=') or arg.startswith('--pretrained='):
            pretrained = arg[arg.index('=') + 1:]
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

    # Init datasets and logger
    ds.init_data()
    log.init(exp_name)
    if not resume:
        train_start_log()

    # Create an instance of the model
    generator = RDN(ds.scale, 3, 64, 64, 16, 8)
    generator.to(device)
    discriminator = ConvDiscr(6, 64)
    discriminator.to(device)

    # Resume from the last checkpoint or load pretrained weights
    if resume:
        PATH = ds.SAVE_DIR + 'model_instances/checkpoint'
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_acc']
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        algorithm.gen_opt_state_dict = checkpoint['gen_optimizer']
        algorithm.dis_opt_state_dict = checkpoint['dis_optimizer']
        use_warmup = False
    elif pretrained is not None:
        PATH = ds.SAVE_DIR + 'model_instances/' + pretrained + '.pth'
        generator.load_state_dict(torch.load(PATH))

    if start_epoch == epoch_count:
        print('Cannot resume training, already reached last epoch!')
        return

    # Train model
    train(generator, discriminator, device, epoch_count=epoch_count, start_epoch=start_epoch,
          use_scheduler=use_scheduler, use_warmup=use_warmup, best_accuracy=best_result)

    # Test model on all valid data
    if ds.valid_set_size != 0:
        ds.valid_set_size = 0
        ds.init_data()
        generator.eval()
        discriminator.eval()
        acc, gen_loss, dis_loss, _ = valid(generator, discriminator, device,
                                           save_images=False, title="Valid Full")
        print('Full valid: GEN loss: %.3f, DIS loss: %.3f' % (gen_loss, dis_loss))
        print('Full valid metric: %.2f\n' % acc)


def start_predict():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python main.py --help" for usage info')
        return

    pretrained = sys.argv[1]

    cuda_id = 0
    for arg in sys.argv[2:]:
        if arg == '--predict':
            pass
        elif arg.startswith('-g=') or arg.startswith('--gpu='):
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
    generator = RDN(ds.scale, 3, 64, 64, 16, 8)
    generator.to(device)

    PATH = ds.SAVE_DIR + 'model_instances/' + pretrained + '.pth'
    generator.load_state_dict(torch.load(PATH))

    # Inference model on images in 'predict' folder
    predict(generator, device)


def start_inference():
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python main.py --help" for usage info')
        return

    pretrained = sys.argv[1]
    length = start = 0
    cuda_id = 0
    for arg in sys.argv[2:]:
        if arg == '-i' or arg == '--inference':
            pass
        elif arg.startswith('-g=') or arg.startswith('--gpu='):
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
    generator = RDN(ds.scale, 3, 64, 64, 16, 8)
    generator.to(device)

    PATH = ds.SAVE_DIR + 'model_instances/' + pretrained + '.pth'
    generator.load_state_dict(torch.load(PATH))

    # Process video in 'video' folder
    inference(generator, device, length, start)


if __name__ == "__main__":
    if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
        print(man.usage)
    elif sys.argv.__contains__('--predict'):
        start_predict()
    elif sys.argv.__contains__('--inference') or sys.argv.__contains__('-i'):
        start_inference()
    else:
        start_train()
