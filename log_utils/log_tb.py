"""Tensorboard log utils"""

from torch.utils.tensorboard import SummaryWriter
from dl_modules.dataset import SAVE_DIR

# Create writer and name scalars
writer = None  # SummaryWriter("log/not_categorized")
scalar_labels = ["Train PSNR", "Train SSIM", "Train LPIPS",
                 "Valid PSNR", "Valid SSIM", "Valid LPIPS",
                 "GEN Train Loss", "DIS Train Loss",
                 "GEN Valid Loss", "DIS Valid Loss",
                 "GEN Learning Rate", "DIS Learning Rate"]
constant_labels = ["Bicubic Acc", "Bicubic Loss"]
image_labels = ["Valid Pred 1", "Valid Pred 2", "Valid Pred 3",
                "Train LR", "Train Pred", "Train HR",
                "Valid LR 1", "Valid HR 1", "Valid LR 2",
                "Valid HR 2", "Valid LR 3", "Valid HR 3"]
constant_value = [None] * len(constant_labels)


def init(exp_id: str) -> None:
    """Initialize logger.
        Args:
            exp_id: name of experiment
        Returns: """
    global writer
    writer = SummaryWriter(SAVE_DIR + "log/" + exp_id)


def save() -> None:
    """Flush log changes"""
    global writer
    writer.flush()


def add(epoch_idx: int, scalars: tuple=None,
                        images: tuple=None,
                        constants: tuple=None,
                        im_start: int=0) -> None:

    """Save values to log.
        Args:
            epoch_idx: current epoch number
            scalars: tuple of scalar values
            images: tuple of PIL / numpy / Tensor images
            constants: tuple of constant values
            im_start: start index for image labels
        Returns: """

    global writer
    if scalars is not None:
        for i in range(len(scalars)):
            writer.add_scalar(scalar_labels[i], scalars[i], epoch_idx)
    if images is not None:
        for i in range(len(images)):
            writer.add_image(image_labels[im_start + i], images[i], epoch_idx)

    for i in range(len(constant_labels)):
        if constant_value[i] is not None:
            writer.add_scalar(constant_labels[i], constant_value[i], epoch_idx)

    if constants is not None:
        for i in range(len(constants)):
            writer.add_scalar(constant_labels[i], constants[i], epoch_idx)
            constant_value[i] = constants[i]
    # print("Epoch", epoch_idx, "added to board log")
