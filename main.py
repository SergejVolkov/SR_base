import io
import sys
import torch
import os.path
import log_utils.log_tensorboard as log
import scripts.dataset as ds
import scripts.algorithm as algorithm
import resources.manual as man
from models.RDN import RDN
from models.SimpleDiscr import ConvDiscr
# from models.Algo import Bicubic
from scripts.train import train
from scripts.validation import valid
from scripts.validation import get_static_images
# from scripts.predict import predict
# from scripts.inference import inference


def train_start_log():
    # Evaluate naive solution for future comparison
    # naive = Bicubic()
    # naive.to(device)
    # naive_acc, naive_loss, _ = valid(naive, device, save_images=False, title="Valid Bicubic")
    # print('Bicubic loss: %.3f, bicubic accuracy: %.3f' % (naive_loss, naive_acc))
    # log.add(epoch_idx=0, constants=(naive_acc, naive_loss))

    # Add static images to log
    log.add(epoch_idx=0, images=tuple(get_static_images()), im_start=6)


if __name__ == "__main__":
    if sys.argv.__contains__('--help') or sys.argv.__contains__('-h'):
        print(man.usage)
        exit(0)
    if len(sys.argv) < 3:
        print('Wrong number of params!\nTry "python main.py --help" for usage info')
        exit(0)

    epoch_count = int(sys.argv[1])
    exp_name = sys.argv[2]
    pretrained = None
    start_epoch = 0
    use_scheduler = use_warmup = True
    resume = False
    for arg in sys.argv[3:]:
        if arg == '-r' or arg == '--resume':
            PATH = ds.SAVE_DIR + 'model_instances/checkpoint.pth'
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

    # Try to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

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
        PATH = ds.SAVE_DIR + 'model_instances/checkpoint.pth'
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint['epoch']
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        algorithm.gen_opt_state_dict = checkpoint['gen_optimizer']
        algorithm.dis_opt_state_dict = checkpoint['dis_optimizer']
    elif pretrained is not None:
        PATH = ds.SAVE_DIR + 'model_instances/' + pretrained + '.pth'
        generator.load_state_dict(torch.load(PATH))

    # Train model
    train(generator, discriminator, device, epoch_count=epoch_count, start_epoch=start_epoch,
          use_scheduler=use_scheduler, use_warmup=use_warmup)

    # Test model on all valid data
    if ds.valid_set_size != 0:
        generator.eval()
        discriminator.eval()
        acc, loss, pred = valid(generator, discriminator, device,
                                save_images=False, title="Valid Full")
        print('Test loss: %.3f, test accuracy: %.3f' % (acc, loss))

    # Inference model on images in 'predict' folder
    # predict(net, device)

    # Process video in 'video' folder
    # inference(net, device, 5, 256)

    # Save our beautiful model for future generations
    # PATH = 'model_instances/cifar_net_tmp.pth'
    # torch.save(net.state_dict(), PATH)
