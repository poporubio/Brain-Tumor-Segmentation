from abc import ABC, abstractmethod

from utils import MetricClass
from .wrappers import (
    AsyncDataGeneratorWrapper,
    NormalizedDataGeneratorWrapper,
    # AugmentedDataGeneratorWrapper,
)


class DataProviderBase(ABC):

    _metric = MetricClass

    def get_testing_data_generator(self, **kwargs):
        return self._get_data_generator(self.test_ids, augmentation=False, **kwargs)

    def get_training_data_generator(self, **kwargs):
        return self._get_data_generator(self.train_ids, augmentation=True, **kwargs)

    def _get_data_generator(self, data_ids, augmentation, async_load=True, **kwargs):
        data_generator = self._get_raw_data_generator(data_ids, **kwargs)
        data_generator = NormalizedDataGeneratorWrapper(data_generator)
        # if augmentation:
        #     data_generator = AugmentedDataGeneratorWrapper(data_generator)
        if async_load:
            data_generator = AsyncDataGeneratorWrapper(data_generator)
        return data_generator

    @abstractmethod
    def _get_raw_data_generator(self, data_ids, **kwargs):
        pass

    @property
    def metric(self):
        return self._metric

    @property
    @abstractmethod
    def data_format(self) -> dict:
        pass
