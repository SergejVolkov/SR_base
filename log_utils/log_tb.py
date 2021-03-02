"""Tensorboard log utils"""

from torch.utils.tensorboard import SummaryWriter
from dl_modules.dataset import SAVE_DIR

# Create writer and name scalars
writer = None  # SummaryWriter("log/not_categorized")
scalar_labels = ["mtr PSNR", "mtr SSIM", "mtr LPIPS",
                 "mval PSNR", "mval SSIM", "mval LPIPS",
                 "ltr GEN", "ltr DIS",
                 "lval GEN", "lval DIS",
                 "lr GEN", "lr DIS"]
constant_labels = ["l Bicubic", "l Bicubic"]
image_labels = ["ival 1 Pred", "ival 3 Pred", "ival 3 Pred",
                "itr LR", "itr Pred", "itr HR",
                "ival 1 LR", "ival 1 HR", "ival 2 LR",
                "ival 2 HR", "ival 3 LR", "ival 3 HR"]
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
