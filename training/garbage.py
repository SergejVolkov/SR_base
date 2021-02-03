def get_training_augmentation():
    crop_size = 64
    return transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=1),
            transforms.ColorJitter(saturation=1),
            transforms.ColorJitter(contrast=1),
            transforms.ColorJitter(hue=1)
        ]), p=0.1),

        transforms.RandomApply(torch.nn.ModuleList([
            transforms.GaussianBlur(kernel_size=3),
        ]), p=0.1),

        transforms.RandomCrop(size=crop_size)
    ])