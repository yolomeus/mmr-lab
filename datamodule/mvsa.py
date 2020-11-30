import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from hydra.utils import to_absolute_path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Normalize, Compose
from tqdm import tqdm

from datamodule import DatasetSplit
from datamodule.default_datamodule import KFoldDataModule
from tokenization.twokenize import fe_net_tokenize


class MVSADataModule(KFoldDataModule):
    """DataModule for the multi-view social data dataset. It assumes the dataset to be pre-processed already. For more
    info about the original dataset: https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
    """

    def __init__(self,
                 data_dir,
                 k_splits,
                 train_conf,
                 test_conf,
                 num_workers,
                 pin_memory):
        super().__init__(k_splits, train_conf, test_conf, num_workers, pin_memory)

        self._current_fold = None
        self.train_ids = None
        self.val_ids = None
        self.test_ids = None
        self.id_to_label = None

        self.word_to_id = None

        self.special_tokens = ['<PAD>', '<UNK>']
        self.label_filename = 'valid_pairlist.txt'
        self.split_dirname = 'splits'

        self.data_dir = to_absolute_path(data_dir)
        self.split_dir = os.path.join(data_dir, self.split_dirname)
        self.instances_dir = os.path.join(data_dir, 'data')

    def set_fold(self, i):
        self._current_fold = i
        split_n = i + 1

        train_file = os.path.join(self.split_dir, f'train_{split_n}.txt')
        val_file = os.path.join(self.split_dir, f'val_{split_n}.txt')
        test_file = os.path.join(self.split_dir, f'test_{split_n}.txt')

        self.train_ids = self._load_ids(train_file)
        self.val_ids = self._load_ids(val_file)
        self.test_ids = self._load_ids(test_file)

    def prepare_data(self, *args, **kwargs):
        # build vocab or load existing
        os.makedirs(to_absolute_path('data/generated/'), exist_ok=True)
        vocab_file = to_absolute_path('data/generated/vocab.json')
        if not os.path.exists(vocab_file):
            self.word_to_id = self._build_vocab(vocab_file)
        else:
            with open(vocab_file, 'r', encoding='utf8') as fp:
                self.word_to_id = json.load(fp)

    def setup(self, stage: Optional[str] = None):
        # load labels
        label_filepath = os.path.join(self.data_dir, self.label_filename)
        id_label = pd.read_csv(label_filepath).to_numpy()
        self.id_to_label = {idx: label for idx, label in id_label}

        # initialize with first fold
        self.set_fold(0)

    def prepare_instances_and_labels(self, split: DatasetSplit):
        if split == DatasetSplit.TRAIN:
            ids = self.train_ids
        elif split == DatasetSplit.VALIDATION:
            ids = self.val_ids
        elif split == DatasetSplit.TEST:
            ids = self.test_ids
        else:
            raise NotImplementedError()

        # filter out contradictory instances, i.e. those without label
        ids = list(filter(lambda x: x in self.id_to_label, ids))
        return self.instances_dir, ids, self.id_to_label, self.word_to_id

    def get_dataset_class(self):
        return MVSADataset

    @staticmethod
    def _load_ids(path):
        with open(path, 'r', encoding='utf8') as id_file:
            ids = id_file.readlines()
        ids = list(map(int, map(str.strip, ids)))
        return ids

    def _build_vocab(self, output_file):
        vocab = set()
        # build vocabulary
        for filename in tqdm(os.listdir(self.instances_dir), desc='building vocabulary'):
            if filename.endswith('.txt'):
                text_path = os.path.join(self.instances_dir, filename)
                with open(text_path, 'r', encoding='utf8') as fp:
                    tweet = fp.readline()
                    tokens = fe_net_tokenize(tweet)
                    for token in tokens:
                        vocab.add(token.lower())

        word_to_id = {word: i for i, word in enumerate(self.special_tokens)}
        for i, word in enumerate(list(vocab)):
            word_to_id[word] = i + len(self.special_tokens)

        with open(output_file, 'w', encoding='utf8') as vocab_file:
            json.dump(word_to_id, vocab_file)

        return word_to_id


class MVSADataset(Dataset):
    def __init__(self, data_dir, ids, id_to_label, word_to_id):
        self.data_dir = data_dir
        self.ids = ids
        self.id_to_label = id_to_label
        self.word_to_id = word_to_id

        resnet_mean = np.array([0.485, 0.456, 0.406])
        resnet_std = np.array([0.229, 0.224, 0.225])
        self.resnet_transform = Sequential(
            Resize((224, 224)),
            Normalize(mean=resnet_mean.tolist(), std=resnet_std.tolist()),
        )
        self.denormalize = Normalize((-resnet_mean / resnet_std).tolist(),
                                     (1.0 / resnet_std).tolist())

    def __getitem__(self, index):
        file_id = self.ids[index]

        img = self._read_img_instance(file_id)
        # self._show_img(img)
        img = self.resnet_transform(img / 255.0)
        # img = (self.denormalize(img) * 255).type(torch.uint8)
        # self._show_img(img)

        text = self._read_text_instance(file_id)
        tokens = fe_net_tokenize(text)
        token_ids = [self.word_to_id.get(token, '<UNK>') for token in tokens]

        label = self.id_to_label[file_id]

        return (img, token_ids), label

    def __len__(self):
        return len(self.ids)

    def _read_img_instance(self, file_id):
        img_path = os.path.join(self.data_dir, f'{file_id}.jpg')
        img = torchvision.io.read_image(img_path)
        return img

    def _read_text_instance(self, file_id):
        text_path = os.path.join(self.data_dir, f'{file_id}.txt')
        with open(text_path, 'r', encoding='utf8') as fp:
            tweet = fp.readline()
        return tweet

    @staticmethod
    def _show_img(img):
        np_img = img.numpy().transpose(1, 2, 0)
        pil_img = Image.fromarray(np_img)
        pil_img.show()
