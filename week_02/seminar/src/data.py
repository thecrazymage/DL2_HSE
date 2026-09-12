import torch
from torchvision import datasets, transforms

# ---------------------------
# Data: CIFAR-10
# ---------------------------

class CIFAR10Optimized(datasets.CIFAR10):
    def __init__(self, root, train = True, transform = None, target_transform = None, download = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        # self.data = [Image.fromarray(img) for img in self.data]
        self.data = torch.from_numpy(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def make_dataloaders(data_dir, batch_size, workers, pin_memory, use_faster_dataset):
    
    if use_faster_dataset:
        # NOTE all transformation will be handled on GPU
        train_set = CIFAR10Optimized(root=data_dir, train=True, download=True, transform=None)
    else:
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)

    dl = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(workers > 0),
    )
    return dl
