"""Training/testing loops specified by pytorch-lightning models. Unlike in standard pytorch-lightning, the loop should
encapsulate the model instead of being bound to it by inheritance. This way, different models can easily be plugged into
the same training procedures.
"""
from abc import ABC

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from datamodule import DatasetSplit
from log.utils import Metrics


class AbstractBaseLoop(LightningModule, ABC):
    """Abstract base class for implementing a training loop for a pytorch model.
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)


class DefaultClassificationLoop(AbstractBaseLoop, ABC):
    """Default wrapper for training/testing a pytorch module using pytorch-lightning. Assumes a standard classification
    task with instance-label pairs (x, y) and a loss function that has the signature loss(y_pred, y_true).
    """

    def __init__(self, hparams: DictConfig):
        """
        :param hparams: contains all hyperparameters.
        """
        super().__init__(hparams)

        self.model = instantiate(hparams.model)
        self.loss = instantiate(hparams.loss)
        self.metrics = Metrics(self.loss, hparams.metrics)

    def configure_optimizers(self):
        return instantiate(self.hparams.optimizer, self.parameters())

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y_true)

        self.log('batch_loss', loss)
        return {'loss': loss, 'y_pred': y_pred, 'y_true': y_true}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        return {'y_pred': y_pred, 'y_true': y_true}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, DatasetSplit.TRAIN)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, DatasetSplit.VALIDATION)

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, DatasetSplit.TEST)

    def _epoch_end(self, outputs, split: DatasetSplit):
        """Compute loss and all metrics at the end of an epoch.

        :param split: prefix for logs e.g. train, test, validation
        :param outputs: gathered outputs from *_epoch_end
        :return: a dict containing loss and metric logs.
        """
        logs = self.metrics.compute_logs(outputs, split)
        self.log_dict(logs)
