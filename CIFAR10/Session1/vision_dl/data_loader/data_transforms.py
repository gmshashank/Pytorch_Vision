import torch
from torchvision import transforms


def data_transforms(augmentation=False, rotation=0.0):
    """Create data transformations

    Args:
        augmentation: Whether to apply data augmentation.
            Defaults to False.
        rotation: Angle of rotation for image augmentation.
            Defaults to 0. It won't be needed if augmentation is False.

    Returns:
        Transform object containing defined data transformations.
    """

    transforms_list = [
        # convert the data to torch.FloatTensor
        # with values within the range [0.0 ,1.0]
        transforms.ToTensor(),
        # normalize the data with mean and standard deviation to keep values in range [-1, 1]
        # since there are 3 channels for each image,
        # we have to specify mean and std for each channel
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    if augmentation:
        transforms_list = [
            # Rotate image by 6 degrees
            transforms.RandomRotation((-rotation, rotation), fill=(1,))
        ] + transforms_list

    return transforms.Compose(transforms_list)
