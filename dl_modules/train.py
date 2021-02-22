import sys
import time
import pyprind
import torch
import torch.nn as nn
import dl_modules.dataset as ds
import dl_modules.algorithm as algorithm
import dl_modules.scheduler as scheduler
import dl_modules.warmup as warmup
import dl_modules.valid as validation
import log_utils.log_tb as log
import torch.nn.functional as F

from datetime import timedelta


# Train model for 'epoch_count' epochs
def train(gen_model: nn.Module, dis_model: nn.Module, device: torch.device,
          epoch_count: int, start_epoch: int=0,
          use_scheduler: bool=False, use_warmup: bool=False) -> None:
    start_time = time.time()
    super_criterion = algorithm.get_super_loss()
    gen_criterion = algorithm.get_gen_loss()
    dis_criterion = algorithm.get_dis_loss()
    gen_opt = algorithm.get_gen_optimizer(gen_model)
    dis_opt = algorithm.get_dis_optimizer(dis_model)
    metric = algorithm.get_metric()

    best_accuracy = 0.0
    epoch_idx = start_epoch

    for epoch_idx in range(start_epoch, epoch_count):
        if use_scheduler:
            algorithm.update_optimizer(gen_opt, scheduler.get_params())

            # Finish training if scheduler thinks it's time
            if not scheduler.active:
                break

        gen_model.train()
        dis_model.train()

        average_gen_loss = 0.0
        average_dis_loss = 0.0
        train_accuracy = 0
        total = len(ds.train_loader)
        scaled_inputs = outputs = gt = None

        iter_bar = pyprind.ProgBar(total, title="Train", stream=sys.stdout)
        
        for sample_id, data in enumerate(ds.train_loader, 0):
            if use_warmup and warmup.active:
                algorithm.update_optimizer(gen_opt,
                                           warmup.get_params(epoch_idx, sample_id))

            inputs, gt = data
            inputs = inputs.to(device)
            gt = gt.to(device)

            gen_opt.zero_grad()
            dis_opt.zero_grad()

            outputs = gen_model(inputs)

            # Conditional GAN (see paper for further explanations)
            scaled_inputs = F.interpolate(
                inputs, scale_factor=(ds.scale, ds.scale), mode='bicubic', align_corners=True
            )
            concat_outputs = torch.cat((outputs, scaled_inputs), 1)
            concat_gt = torch.cat((gt, scaled_inputs), 1)

            # Generator step
            dis_model.requires_grad(False)

            gen_loss = super_criterion(outputs, gt) + \
                       gen_criterion(dis_model(concat_outputs), dis_model(concat_gt))

            # Discriminator step
            dis_model.requires_grad(True)
            concat_outputs = concat_outputs.detach()

            dis_loss = dis_criterion(dis_model(concat_outputs), dis_model(concat_gt))

            # Compute gradients
            gen_loss.backward()
            dis_loss.backward()

            # Perform weights update
            gen_opt.step()
            dis_opt.step()

            # Gather stats
            average_gen_loss += gen_loss.item()
            average_dis_loss += dis_loss.item()
            train_accuracy += metric(outputs, gt).item()

            iter_bar.update()

        iter_bar.update()

        average_gen_loss /= total
        average_dis_loss /= total
        train_accuracy /= total

        gen_model.eval()
        dis_model.eval()
        valid_accuracy, valid_gen_loss, valid_dis_loss, images = \
            validation.valid(gen_model, dis_model, device, save_images=True)

        if use_scheduler:
            scheduler.add_metrics(valid_accuracy)
            gen_lr = scheduler.gen_lr
            dis_lr = scheduler.dis_lr
        else:
            gen_lr = gen_opt.param_groups[0]['lr']
            dis_lr = dis_opt.param_groups[0]['lr']
        if use_warmup and warmup.active:
            gen_lr = warmup.gen_lr
            dis_lr = warmup.dis_lr

        # Print useful numbers
        print('Epoch %3d:\nTrain:  GEN lr: %g, DIS lr: %g\n'
              '       GEN loss: %.3f, DIS loss: %.3f\n'
              'Valid: GEN loss: %.3f, DIS loss: %.3f' %
              (epoch_idx, gen_lr, dis_lr, average_gen_loss, average_dis_loss, valid_gen_loss, valid_dis_loss))
        print('Train metric: %.2f\n'
              'Valid metric: %.2f\n' % (train_accuracy, valid_accuracy))

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            PATH = ds.SAVE_DIR + 'model_instances/gen_epoch_%d_acc_%.2f.pth' % (epoch_idx, valid_accuracy)
            torch.save(gen_model.state_dict(), PATH)
            PATH = ds.SAVE_DIR + 'model_instances/dis_epoch_%d_acc_%.2f.pth' % (epoch_idx, valid_accuracy)
            torch.save(dis_model.state_dict(), PATH)

        checkpoint = {
            'epoch': epoch_idx,
            'generator': gen_model.state_dict(),
            'discriminator': dis_model.state_dict(),
            'gen_optimizer': gen_opt.state_dict(),
            'dis_optimizer': dis_opt.state_dict()
        }
        torch.save(checkpoint, ds.SAVE_DIR + 'model_instances/checkpoint.pth')

        # Prepare train samples for export
        inputs = torch.clamp(scaled_inputs[0, :, :, :] / 2 + 0.5, min=0, max=1)
        outputs = torch.clamp(outputs[0, :, :, :] / 2 + 0.5, min=0, max=1)
        gt = torch.clamp(gt[0, :, :, :] / 2 + 0.5, min=0, max=1)

        # Save log
        log.add(epoch_idx=epoch_idx,
                scalars=(train_accuracy, valid_accuracy, average_gen_loss, average_dis_loss,
                         valid_gen_loss, valid_dis_loss, gen_lr, dis_lr),
                images=tuple(images + [inputs, outputs, gt]))
        log.save()

    # Finish training
    total_time = int(time.time() - start_time)
    print('Complete!\n')
    print('Average epoch train time:', str(timedelta(
        seconds=total_time // (epoch_idx - start_epoch))))
    print('Total time:', str(timedelta(seconds=total_time)))
