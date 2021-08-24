import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all dataloaders
    """

    def __init__(
        self, dataset, batch_size, shuffle, validation_split, num_workers, pin_memory
    ):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.batch_idx = 0
        self.num_samples = len(dataset)
        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split
        )

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        super().__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        idx_full = np.arange(self.num_samples)
        np.random.seed(42)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.num_samples, "Validation set larger than entire dataset"
            len_valid = split
        else:
            len_valid = int(self.num_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))  # idx_full[:len_valid]

        # samplers
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle,mutually exclusive with sampler
        self.shuffle = False
        self.num_samples = len(train_idx)
        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
