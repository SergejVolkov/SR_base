"""Tensorboard log utils"""

from torch.utils.tensorboard import SummaryWriter

# Create writer and name scalars
writer = None  # SummaryWriter("log/not_categorized")
scalar_labels = ["Train Acc", "Test Acc", "Train Loss", "Test Loss", 'Learning Rate']
image_labels = ["Prediction"]


def init(exp_id: str) -> None:
    """Initialize logger.
        Args:
            exp_id: name of experiment
        Returns: """
    global writer
    writer = SummaryWriter("../drive/MyDrive/log/" + exp_id)


def save() -> None:
    """Flush log changes"""
    global writer
    writer.flush()


def add(epoch_idx: int, scalars: tuple=None, images: tuple=None) -> None:
    """Save values to log.
        Args:
            epoch_idx: current epoch number
            scalars: tuple of scalar values
            images: tuple of PIL images
        Returns: """
    global writer
    if scalars is not None:
        for i in range(len(scalars)):
            writer.add_scalar(scalar_labels[i], scalars[i], epoch_idx)
    if images is not None:
        for i in range(len(images)):
            if images[i] is not None:
                writer.add_image(image_labels[i], images[i], epoch_idx)
    print("Epoch", epoch_idx, "added to board log")
