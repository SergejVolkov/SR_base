import torch
import log_utils.log_tensorboard as log
import training.scheduler as scheduler
import training.dataset as ds
# from models.ResNet import sr_resnet18
from models.RDN import RDN
from training.train import train
# from training.predict import predict


if __name__ == "__main__":
    log.init("rdn")

    # Try to use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Create an instance of the model
    net = RDN(ds.scale, 3, 64, 64, 16, 8)
    net.to(device)

    PATH = '../drive/MyDrive/model_instances/net_tmp_epoch_28_acc_27.67.pth'
    net.load_state_dict(torch.load(PATH))

    train(net, epoch_count=scheduler.count_epoch(), start_epoch=0, use_scheduler=True)
    # predict(net)

    # Save our beautiful model for future generations
    # PATH = 'model_instances/cifar_net_tmp.pth'
    # torch.save(net.state_dict(), PATH)
