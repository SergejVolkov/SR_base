import sys
import time
import pyprind
import torch
import torch.nn as nn
import training.dataset as ds
import training.algorithm as algorithm
import training.scheduler as scheduler
import training.validation as validation
import log_utils.log_tensorboard as log

from datetime import timedelta


# Train model for 'epoch_count' epochs
def train(net: nn.Module, epoch_count: int, start_epoch: int=0,
          use_scheduler: bool=False) -> None:
    start_time = time.time()
    criterion = algorithm.get_loss()
    metric = algorithm.get_metric()
    if use_scheduler:
        optimizer = algorithm.get_optimizer(net, scheduler.params_list[start_epoch])
    else:
        optimizer = algorithm.get_optimizer(net)

    best_accuracy = 0.0

    for epoch_idx in range(start_epoch, epoch_count):
        net.train()

        if use_scheduler:
            algorithm.update_optimizer(optimizer, scheduler.params_list[epoch_idx])

        average_loss = 0.0
        train_accuracy = 0
        total = len(ds.train_loader)

        iter_bar = pyprind.ProgBar(total, title="Train", stream=sys.stdout)
        
        for _, data in enumerate(ds.train_loader, 0):
            inputs, gt = data
            inputs = inputs.cuda()
            gt = gt.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

            average_loss += loss.item()
            train_accuracy += metric(outputs, gt).item()

            iter_bar.update()

        iter_bar.update()

        average_loss /= total
        train_accuracy /= total

        net.eval()
        test_accuracy, test_loss, prediction = validation.valid(net)

        # Print useful numbers
        print('[%d, %5d] train loss: %.3f, test loss: %.3f' %
              (epoch_idx, total, average_loss, test_loss))
        print('Train accuracy: %.2f %%' % train_accuracy)
        print('Test accuracy: %.2f %%' % test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            PATH = '../drive/MyDrive/model_instances/net_tmp_epoch_%d_acc_%.2f.pth' % (epoch_idx, test_accuracy)
            torch.save(net.state_dict(), PATH)
            print('Model saved!')

        log.add(epoch_idx, (train_accuracy, test_accuracy,
                            average_loss, test_loss, scheduler.params_list[epoch_idx][0]),
                (prediction, ))
        log.save()
    total_time = int(time.time() - start_time)
    print('Complete!\n')
    print('Average epoch train time:', str(timedelta(
        seconds=total_time // (epoch_count - start_epoch))))
    print('Total time:', str(timedelta(seconds=total_time)))
