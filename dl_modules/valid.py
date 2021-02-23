# import sys
# import pyprind
import torch
import dl_modules.dataset as ds
import dl_modules.algorithm as algorithm
import torch.nn.functional as F


images_to_save = 3


def valid(gen_model: torch.nn.Module, dis_model: torch.nn.Module, device: torch.device,
          save_images=False, title="Valid") -> (int, float, list):
    super_criterion = algorithm.get_super_loss()
    gen_criterion = algorithm.get_gen_loss()
    dis_criterion = algorithm.get_dis_loss()
    metric = algorithm.get_metric()
    average_gen_loss = 0.0
    average_dis_loss = 0.0
    accuracy = 0.0
    total = len(ds.valid_loader)

    # iter_bar = pyprind.ProgBar(total, title=title, stream=sys.stdout)
    images = []

    with torch.no_grad():
        for data in ds.valid_loader:
            inputs, gt = data
            inputs = inputs.to(device)
            gt = gt.to(device)

            outputs = gen_model(inputs)

            scaled_inputs = F.interpolate(
                inputs, scale_factor=(ds.scale, ds.scale), mode='bicubic', align_corners=True
            )
            concat_outputs = torch.cat((outputs, scaled_inputs), 1)
            concat_gt = torch.cat((gt, scaled_inputs), 1)

            gen_loss = super_criterion(outputs, gt) + algorithm.gan_loss_coeff * \
                       gen_criterion(dis_model(concat_outputs), dis_model(concat_gt))
            dis_loss = dis_criterion(dis_model(concat_outputs), dis_model(concat_gt))

            average_gen_loss += gen_loss.item()
            average_dis_loss += dis_loss.item()
            accuracy += metric(outputs, gt).item()

            if save_images and len(images) < images_to_save:
                images.append(
                    torch.clamp(outputs.squeeze(0) / 2 + 0.5, min=0, max=1)
                )

            # iter_bar.update()
    # iter_bar.update()
    return accuracy / total, average_gen_loss / total, average_dis_loss / total, images


def get_static_images() -> list:
    images = []

    for data in ds.valid_loader:
        inputs, gt = data
        # Add LR sample
        images.append(
            torch.clamp(F.interpolate(
                inputs, scale_factor=(ds.scale, ds.scale), mode='bicubic', align_corners=True
            ).squeeze(0) / 2 + 0.5, min=0, max=1)
        )
        # Add HR sample
        images.append(torch.clamp(gt.squeeze(0) / 2 + 0.5, min=0, max=1))
        if len(images) >= images_to_save * 2:
            break

    return images
