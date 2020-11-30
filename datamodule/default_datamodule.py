from abc import abstractmethod, ABC
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from datamodule import DatasetSplit


class AbstractDefaultDataModule(LightningDataModule):
    """Base class for pytorch-lightning DataModule datasets. Subclass this if you have a standard train, validation,
    test split.
    """

    def __init__(self, train_conf, test_conf, num_workers, pin_memory, collate_fn_name='collate_fn'):
        """

        :param train_conf:
        :param test_conf:
        :param num_workers:
        :param pin_memory:
        :param collate_fn_name:
        """
        super().__init__()

        self._train_conf = train_conf
        self._test_conf = test_conf
        self._num_workers = num_workers
        self._pin_memory = pin_memory

        self._collate_fn_name = collate_fn_name

    @property
    @abstractmethod
    def train_ds(self):
        """Build the train pytorch dataset.

        :return: the train pytorch dataset.
        """

    @property
    @abstractmethod
    def val_ds(self):
        """Build the validation pytorch dataset.

        :return: the validation pytorch dataset.
        """

    @property
    @abstractmethod
    def test_ds(self):
        """Build the test pytorch dataset.

        :return: the test pytorch dataset.
        """

    def train_dataloader(self):
        collate_fn = getattr(self.train_ds, self._collate_fn_name, None)
        train_dl = DataLoader(self.train_ds,
                              self._train_conf.batch_size,
                              shuffle=True,
                              num_workers=self._num_workers,
                              pin_memory=self._pin_memory,
                              collate_fn=collate_fn)
        return train_dl

    def val_dataloader(self):
        collate_fn = getattr(self.val_ds, self._collate_fn_name, None)
        val_dl = DataLoader(self.val_ds,
                            self._test_conf.batch_size,
                            num_workers=self._num_workers,
                            pin_memory=self._pin_memory,
                            collate_fn=collate_fn)
        return val_dl

    def test_dataloader(self):
        collate_fn = getattr(self.test_ds, self._collate_fn_name, None)
        test_dl = DataLoader(self.test_ds,
                             self._test_conf.batch_size,
                             num_workers=self._num_workers,
                             pin_memory=self._pin_memory,
                             collate_fn=collate_fn)
        return test_dl


class ClassificationDataModule(AbstractDefaultDataModule):
    """Datamodule for a standard classification setting with training instances and labels.
    """

    @abstractmethod
    def prepare_instances_and_labels(self, split: DatasetSplit):
        """Get tuple of instances and labels for classification.

        :param split: the dataset split use.
        :return (tuple): instances and labels for a specific split.
        """

    def get_dataset_class(self):
        """This method can be overwritten to replace the standard classification dataset class.

        :return: a torch.utils.Dataset class to be used.
        """
        return self._ClassificationDataset

    def _create_dataset(self, split: DatasetSplit):
        """Helper factory method for building a split specific pytorch dataset.

        :param split: which split to build.
        :return: the split specific pytorch dataset.
        """
        return self.get_dataset_class()(*self.prepare_instances_and_labels(split))

    @property
    def train_ds(self):
        return self._create_dataset(DatasetSplit.TRAIN)

    @property
    def val_ds(self):
        return self._create_dataset(DatasetSplit.VALIDATION)

    @property
    def test_ds(self):
        return self._create_dataset(DatasetSplit.TEST)

    class _ClassificationDataset(Dataset):
        """Pytorch Dataset for standard classification setting.
        """

        def __init__(self, instances, labels):
            self.instances = instances
            self.labels = labels

        def __getitem__(self, index):
            return self.instances[index], self.labels[index]

        def __len__(self):
            return len(self.instances)


class KFoldDataModule(ClassificationDataModule):
    """DataModule that needs to select a split when instantiating."""

    def __init__(self, k_folds, train_conf, test_conf, num_workers, pin_memory):
        super().__init__(train_conf, test_conf, num_workers, pin_memory)
        self.k_folds = k_folds

    @abstractmethod
    def setup(self, stage: Optional[str] = None, fold: int = None):
        """Supercharges original setup with fold argument. It should be used as setter, for loading the state of the
        corresponding fold. Doing this through the setup method ensures that the current fold state is synced across
        multiple devices.

        :param stage: see LightningDataModule
        :param fold: the fold to set the state of the DataModule to
        """
