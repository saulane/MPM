import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import segm.utils.torch as ptu


class Loader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, distributed, split, collate_fn=None):
        is_train_split = split in {"train", "trainval"}
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=is_train_split)
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                persistent_workers=(num_workers>0),
                prefetch_factor=4
            )
        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=is_train_split,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                persistent_workers=(num_workers>0),
                prefetch_factor=4
            )

        self.base_dataset = self.dataset

    @property
    def unwrapped(self):
        return self.base_dataset.unwrapped

    def set_epoch(self, epoch):
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

    def get_diagnostics(self, logger):
        return self.base_dataset.get_diagnostics(logger)

    def get_snapshot(self):
        return self.base_dataset.get_snapshot()

    def end_epoch(self, epoch):
        return self.base_dataset.end_epoch(epoch)
