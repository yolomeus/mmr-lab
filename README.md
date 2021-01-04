# Fusion-Extraction Network Pytorch

PyTorch + [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) implementation of
["Fusion-Extraction Network for Multimodal Sentiment Analysis"](
https://link.springer.com/chapter/10.1007/978-3-030-47436-2_59) for K-fold cross validation on the MVSA dataset. The
project uses [hydra](https://github.com/facebookresearch/hydra) for hyperparameter configuration. To pre-process the
dataset and start training run:

```shell
$ python train_kfold.py datamodule.data_dir=path/to/k-fold_mvsa
```

Hyperparameters can be either changed through the hydra config files in ``./conf`` (you can also change the path to the
dataset here) or passed as arguments, e.g.:

```shell
$ python train_kfold.py num_workers=4 training.batch_size=8 training.acc_batches=4
```

### Dependencies

All dependencies can be found in ``environment.yml``. You can use [anaconda](https://www.anaconda.com/) to automatically
create a virtual environment which fulfills the requirements:

```shell
$ conda env create -f environment.yml
```

Or install the dependencies manually using pip.
