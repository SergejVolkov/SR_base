import sys
import time
import pyprind
import torch
import torch.nn as nn
import scripts.dataset as ds
import scripts.algorithm as algorithm
import scripts.scheduler as scheduler
import scripts.warmup as warmup
import scripts.validation as validation
import log_utils.log_tensorboard as log
import torch.nn.functional as F

from datetime import timedelta


# Train model for 'epoch_count' epochs
def train(net: nn.Module, epoch_count: int, start_epoch: int=0,
          use_scheduler: bool=False, use_warmup: bool=False) -> None:
    start_time = time.time()
    criterion = algorithm.get_loss()
    metric = algorithm.get_metric()
    optimizer = algorithm.get_optimizer(net)

    best_accuracy = 0.0

    for epoch_idx in range(start_epoch, epoch_count):
        if use_scheduler:
            algorithm.update_optimizer(optimizer, scheduler.get_params())

        # Finish training if scheduler thinks it's time
        if not scheduler.active:
            break

        net.train()

        average_loss = 0.0
        train_accuracy = 0
        total = len(ds.train_loader)

        iter_bar = pyprind.ProgBar(total, title="Train", stream=sys.stdout)
        
        for sample_id, data in enumerate(ds.train_loader, 0):
            if use_warmup and warmup.active:
                algorithm.update_optimizer(optimizer,
                                           warmup.get_params(epoch_idx, sample_id))

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
        test_accuracy, test_loss, images = validation.valid(net, save_images=True)
        scheduler.add_metrics(test_accuracy)

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

        # Prepare train samples for export
        inputs = torch.clamp(F.interpolate(
            inputs[0, :, :, :].unsqueeze(0), scale_factor=(2, 2), mode='bicubic'
        ).squeeze(0) / 2 + 0.5, min=0, max=1)
        outputs = torch.clamp(outputs[0, :, :, :] / 2 + 0.5, min=0, max=1)
        gt = torch.clamp(gt[0, :, :, :] / 2 + 0.5, min=0, max=1)

        # Save log
        log.add(epoch_idx=epoch_idx,
                scalars=(train_accuracy, test_accuracy,
                         average_loss, test_loss, scheduler.lr),
                images=tuple(images + [inputs, outputs, gt]))
        log.save()

    # Finish training
    total_time = int(time.time() - start_time)
    print('Complete!\n')
    print('Average epoch train time:', str(timedelta(
        seconds=total_time // (epoch_count - start_epoch))))
    print('Total time:', str(timedelta(seconds=total_time)))
