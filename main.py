import torch
import log_utils.log_tensorboard as log
import scripts.dataset as ds
# from models.ResNet import sr_resnet18
from models.RDN import RDN
from models.Algo import Bicubic
from scripts.train import train
from scripts.validation import valid
from scripts.validation import get_static_images
# from scripts.predict import predict


if __name__ == "__main__":
    log.init("rdn")

    # Try to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Evaluate algorithmic method for future comparison
    naive = Bicubic()
    naive.to(device)
    naive_acc, naive_loss, _ = valid(naive, title="Valid Bicubic")
    print('Bicubic loss: %.3f, bicubic accuracy: %.3f' % (naive_loss, naive_acc))
    log.add(epoch_idx=0, constants=(naive_acc, naive_loss))

    # Add static images to log
    log.add(epoch_idx=0, images=tuple(get_static_images()), im_start=6)

    # Create an instance of the model
    net = RDN(ds.scale, 3, 64, 64, 16, 8)
    # net = sr_resnet18()
    net.to(device)

    # Load pretrained weights
    # PATH = '../drive/MyDrive/model_instances/RDN_27.67_(32.35_x1)_L1_x0.5.pth'
    # net.load_state_dict(torch.load(PATH))

    # Train model
    train(net, epoch_count=100, start_epoch=0, use_scheduler=True, use_warmup=True)

    # Test model on raw data
    # ds.valid_set.in_aug = None
    # net.eval()
    # acc, loss, pred = valid(net, title="Valid not aug")
    # print('Test loss: %.3f, test accuracy: %.3f' % (acc, loss))

    # Inference model on images in 'predict' folder
    # predict(net)

    # Save our beautiful model for future generations
    # PATH = 'model_instances/cifar_net_tmp.pth'
    # torch.save(net.state_dict(), PATH)
