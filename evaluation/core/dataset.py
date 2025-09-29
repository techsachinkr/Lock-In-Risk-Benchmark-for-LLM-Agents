"""Dataset implementation for lock-in risk evaluation"""

from typing import List, Optional
from inspect_ai.dataset import Dataset as BaseDataset, Sample

class Dataset(BaseDataset):
    """Dataset implementation for lock-in risk evaluation"""

    def __init__(self, samples: Optional[List[Sample]] = None):
        self._samples = samples or []
        self._name = "lock_in_risk_dataset"
        self._location = None

    def __getitem__(self, index):
        return self._samples[index]

    def __len__(self):
        return len(self._samples)

    @property
    def name(self):
        return self._name

    @property
    def location(self):
        return self._location

    def filter(self, predicate):
        filtered = [s for s in self._samples if predicate(s)]
        new_dataset = Dataset(filtered)
        return new_dataset

    def shuffle(self, seed=None):
        import random
        if seed is not None:
            random.seed(seed)
        random.shuffle(self._samples)
        return self

    def shuffle_choices(self, seed=None):
        # Not needed for our use case
        return self

    def shuffled(self, seed=None):
        new_dataset = Dataset(self._samples[:])
        new_dataset.shuffle(seed)
        return new_dataset

    def sort(self, key=None):
        if key is None:
            self._samples.sort()
        else:
            self._samples.sort(key=key)
        return self

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = value
