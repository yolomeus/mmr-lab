import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from datamodule.default_datamodule import KFoldDataModule
from log.loggers import KFoldWandbLogger


@hydra.main(config_path='conf', config_name='config')
def train_kfold(cfg: DictConfig):
    """Train a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    train_cfg = cfg.training

    logger = KFoldWandbLogger(tags=['train']) if cfg.wandb_log else True

    datamodule: KFoldDataModule = instantiate(cfg.datamodule,
                                              train_conf=cfg.training,
                                              test_conf=cfg.testing,
                                              num_workers=cfg.num_workers,
                                              pin_memory=cfg.gpus > 0)

    # make sure data is prepared before calling setup
    datamodule.prepare_data()
    for i in range(datamodule.k_folds):
        datamodule.setup(fold=i)
        logger.set_fold(i)

        training_loop = instantiate(cfg.loop, cfg)
        logger.watch(training_loop)
        # fold specific checkpoint path
        ckpt_path = os.path.join(os.getcwd(),
                                 f'checkpoints/fold_{i}/',
                                 '{epoch:03d}-{' + train_cfg.monitor + ':.3f}')

        model_checkpoint = ModelCheckpoint(save_top_k=train_cfg.save_ckpts,
                                           monitor=train_cfg.monitor,
                                           mode=train_cfg.mode,
                                           verbose=True,
                                           filepath=ckpt_path)

        early_stopping = EarlyStopping(monitor=train_cfg.monitor,
                                       patience=train_cfg.patience,
                                       mode=train_cfg.mode,
                                       verbose=True)

        trainer = Trainer(max_epochs=train_cfg.epochs,
                          gpus=cfg.gpus,
                          deterministic=True,
                          logger=logger,
                          callbacks=[model_checkpoint, early_stopping],
                          accumulate_grad_batches=train_cfg.acc_batches)

        trainer.fit(training_loop, datamodule=datamodule)

    logger.log_model_average()


if __name__ == '__main__':
    train_kfold()
