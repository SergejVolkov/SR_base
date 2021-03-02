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
from dl_modules.metric.psnr import PSNR
from dl_modules.metric.ssim import SSIM
from lpips_pytorch import LPIPS
from datetime import timedelta


# Train model for 'epoch_count' epochs
def train(gen_model: nn.Module, dis_model: nn.Module, device: torch.device,
          epoch_count: int, start_epoch: int=0, use_scheduler: bool=False,
          use_warmup: bool=False, best_accuracy: float=float('inf'), bars: bool=False) -> None:
    start_time = time.time()
    super_criterion = algorithm.get_super_loss()
    gen_criterion = algorithm.get_gen_loss()
    dis_criterion = algorithm.get_dis_loss()
    gen_opt = algorithm.get_gen_optimizer(gen_model)
    dis_opt = algorithm.get_dis_optimizer(dis_model)
    psnr = PSNR()
    ssim = SSIM()
    lpips = LPIPS()

    epoch_idx = start_epoch
    if use_warmup:
        warmup.init()

    for epoch_idx in range(start_epoch, epoch_count):
        if use_scheduler and not (use_warmup and warmup.active):
            algorithm.update_optimizer(gen_opt, scheduler.get_params_smooth())
            # Finish training if scheduler thinks it's time
            if not scheduler.active:
                break

        gen_model.train()
        dis_model.train()

        average_gen_loss = 0.0
        average_dis_loss = 0.0
        train_psnr = train_ssim = train_lpips = 0.0
        total = len(ds.train_loader)
        scaled_inputs = outputs = gt = None

        if bars:
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

            gen_loss = super_criterion(outputs, gt) + algorithm.gan_loss_coeff * \
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
            norm_out = torch.clamp(outputs.data / 2 + 0.5, min=0, max=1)
            norm_gt = torch.clamp(gt.data / 2 + 0.5, min=0, max=1)
            train_psnr += psnr(norm_out, norm_gt).item()
            train_ssim += ssim(norm_out, norm_gt).item()
            train_lpips += lpips(norm_out, norm_gt).item()

            if bars:
                iter_bar.update()

        average_gen_loss /= total
        average_dis_loss /= total
        train_psnr /= total
        train_ssim /= total
        train_lpips /= total

        # Eval model
        gen_model.eval()
        dis_model.eval()
        valid_psnr, valid_ssim, valid_lpips, valid_gen_loss, valid_dis_loss, images = \
            validation.valid(gen_model, dis_model, device, save_images=True)

        # Get lr
        dis_lr = dis_opt.param_groups[0]['lr']
        if use_warmup and warmup.active:
            gen_lr = warmup.gen_lr
        elif use_scheduler:
            scheduler.add_metrics(valid_gen_loss)
            gen_lr = scheduler.gen_lr
        else:
            gen_lr = gen_opt.param_groups[0]['lr']

        # Print useful numbers
        print('Epoch %3d:\n'
              'Train: GEN lr: %g, DIS lr: %g\n'
              '       GEN loss: %.3f, DIS loss: %.3f\n'
              '       PSNR: %.2f, SSIM: %.4f, LPIPS: %.4f'
              'Valid: GEN loss: %.3f, DIS loss: %.3f'
              '       PSNR: %.2f, SSIM: %.4f, LPIPS: %.4f' %
              (epoch_idx, gen_lr, dis_lr,
               average_gen_loss, average_dis_loss,
               train_psnr, train_ssim, train_lpips,
               valid_gen_loss, valid_dis_loss,
               valid_psnr, valid_ssim, valid_lpips))

        # Save model is better results
        if valid_gen_loss < best_accuracy:
            best_accuracy = valid_gen_loss
            PATH = ds.SAVE_DIR + 'weights/gen_epoch_%d_loss_%.3f.pth' % (epoch_idx, valid_gen_loss)
            torch.save(gen_model.state_dict(), PATH)
            PATH = ds.SAVE_DIR + 'weights/dis_epoch_%d_loss_%.3f.pth' % (epoch_idx, valid_gen_loss)
            torch.save(dis_model.state_dict(), PATH)
            print('Model saved!')
        print('\n', end='')

        # Save checkpoint
        checkpoint = {
            'epoch': epoch_idx,
            'best_acc': best_accuracy,
            'generator': gen_model.state_dict(),
            'discriminator': dis_model.state_dict(),
            'gen_optimizer': gen_opt.state_dict(),
            'dis_optimizer': dis_opt.state_dict()
        }
        torch.save(checkpoint, ds.SAVE_DIR + 'weights/checkpoint')

        # Prepare train samples for export
        inputs = torch.clamp(scaled_inputs[0, :, :, :] / 2 + 0.5, min=0, max=1)
        outputs = torch.clamp(outputs[0, :, :, :] / 2 + 0.5, min=0, max=1)
        gt = torch.clamp(gt[0, :, :, :] / 2 + 0.5, min=0, max=1)

        # Save log
        log.add(epoch_idx=epoch_idx,
                scalars=(train_psnr, train_ssim, train_lpips,
                         valid_psnr, valid_ssim, valid_lpips,
                         average_gen_loss, average_dis_loss,
                         valid_gen_loss, valid_dis_loss,
                         gen_lr, dis_lr),
                images=tuple(images + [inputs, outputs, gt]))
        log.save()

    # Finish training
    total_time = int(time.time() - start_time)
    print('Complete!\n')
    print('Average epoch train time:', str(timedelta(
        seconds=total_time // (epoch_idx - start_epoch))))
    print('Total time:', str(timedelta(seconds=total_time)))
