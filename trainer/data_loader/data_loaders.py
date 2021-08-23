from base import BaseDataLoader
from torchvision import datasets, transforms


class Loader(BaseDataLoader):
    def __init__(
        self, data_dir, batch_size, shuffle=True, validation_split=0.2, num_workers=4
    ):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )