import json
import os
import pickle
from collections import defaultdict
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
from transformers import AutoTokenizer, AutoModel

from datamodule import DatasetSplit
from datamodule.default_datamodule import KFoldDataModule
from tokenization.twokenize import fe_net_tokenize


class MVSADataModule(KFoldDataModule):
    """DataModule for the multi-view social data dataset. It assumes the dataset to be pre-processed already. For more
    info about the original dataset: https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
    """

    def __init__(self,
                 input_dir,
                 output_dir,
                 word_vec_type,
                 k_folds,
                 train_conf,
                 test_conf,
                 num_workers,
                 pin_memory):
        """

        :param input_dir: the MVSA base directory
        :param output_dir: any generated data will be placed here
        :param word_vec_type: type of word embeddings used, can be either 'glove' or 'bert'.
        :param k_folds: total number of folds.
        :param train_conf:
        :param test_conf:
        :param num_workers:
        :param pin_memory:
        """
        super().__init__(k_folds, train_conf, test_conf, num_workers, pin_memory)

        self.word_vec_type = word_vec_type
        self.label_filename = 'valid_pairlist.txt'

        self.input_dir = to_absolute_path(input_dir)
        self.output_dir = to_absolute_path(output_dir)

        self.raw_data_dir = os.path.join(self.input_dir, 'data')
        self.split_dir = os.path.join(self.input_dir, 'splits')

        # here we store state w.r.t the current fold
        self._current_fold = None

        self.train_ids = None
        self.val_ids = None
        self.test_ids = None
        self.id_to_label = None

        # these only apply to glove
        self.special_tokens = ['<PAD>', '<UNK>']
        self.vocab_file = to_absolute_path(os.path.join(output_dir, 'vocab.json'))
        self.word_to_id = None

    def _set_fold(self, i):
        self._current_fold = i
        split_n = i + 1

        train_file = os.path.join(self.split_dir, f'train_{split_n}.txt')
        val_file = os.path.join(self.split_dir, f'val_{split_n}.txt')
        test_file = os.path.join(self.split_dir, f'test_{split_n}.txt')

        self.train_ids = self._load_ids(train_file)
        self.val_ids = self._load_ids(val_file)
        self.test_ids = self._load_ids(test_file)

    def prepare_data(self, *args, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.word_vec_type == 'glove':
            self._build_glove_vocab()
        elif self.word_vec_type == 'bert':
            self._export_bert_embeds()

    def setup(self, stage: Optional[str] = None, fold=None):
        if fold is not None:
            self._set_fold(fold)
        # load labels
        label_filepath = os.path.join(self.input_dir, self.label_filename)
        id_label = pd.read_csv(label_filepath).to_numpy()
        self.id_to_label = {idx: label for idx, label in id_label}
        # load vocab
        with open(self.vocab_file, 'r', encoding='utf8') as fp:
            self.word_to_id = json.load(fp)

        # broken image
        del self.id_to_label[15324]

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
        return self.raw_data_dir, self.output_dir, ids, self.id_to_label, self.word_to_id

    def get_dataset_class(self):
        return MVSADataset if self.word_vec_type == 'glove' else BERTMVSADataset

    @staticmethod
    def _load_ids(path):
        with open(path, 'r', encoding='utf8') as id_file:
            ids = id_file.readlines()
        ids = list(map(int, map(str.strip, ids)))
        return ids

    def _build_glove_vocab(self):
        """Build a vocabulary to use with GloVe and store it in a json file.
        """
        if not os.path.exists(self.vocab_file):
            vocab = defaultdict(int)
            # build vocabulary
            for filename in tqdm(os.listdir(self.raw_data_dir), desc='building vocabulary'):
                if filename.endswith('.txt'):
                    text_path = os.path.join(self.raw_data_dir, filename)
                    with open(text_path, 'r', encoding='utf8', errors='replace') as fp:
                        tweet = fp.readline()
                        tokens = fe_net_tokenize(tweet)
                        for token in tokens:
                            vocab[token.lower()] += 1

            vocab, _ = zip(*filter(lambda x: x[1] > 2, vocab.items()))
            word_to_id = {word: i for i, word in enumerate(self.special_tokens)}
            for i, word in enumerate(list(vocab)):
                word_to_id[word] = i + len(self.special_tokens)

            with open(self.vocab_file, 'w', encoding='utf8') as vocab_file:
                json.dump(word_to_id, vocab_file)

    def _export_bert_embeds(self):
        """Convert text file data to bert embeddings and store on disk.
        """

        cuda_available = torch.cuda.is_available()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert = AutoModel.from_pretrained("bert-base-uncased")
        bert = bert if not cuda_available else bert.to('cuda')

        vec_dict = {}
        text_files = list(filter(lambda x: x.endswith('.txt') and 'c' not in x, os.listdir(self.raw_data_dir)))
        for filename in tqdm(text_files, desc='extracting BERT embeddings'):
            text_path = os.path.join(self.raw_data_dir, filename)
            idx = int(filename.split('.')[0])
            with open(text_path, 'r', encoding='utf8', errors='replace') as fp:
                tweet = fp.readline()
                tokens = tokenizer(tweet, return_tensors="pt")
                tokens = tokens if not cuda_available else tokens.to('cuda')
                with torch.no_grad():
                    vec_seq = bert(**tokens).last_hidden_state.squeeze(0)
                    vec_dict[idx] = vec_seq.detach().cpu()

        for idx, bert_embeds in vec_dict.items():
            out_file = os.path.join(self.output_dir, f'{idx}.pkl')
            with open(out_file, 'wb') as fp:
                pickle.dump(bert_embeds, fp)


class MVSADataset(Dataset):
    def __init__(self, data_dir, output_dir, ids, id_to_label, word_to_id):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.ids = ids
        self.id_to_label = id_to_label
        self.word_to_id = word_to_id

        resnet_mean = np.array([0.485, 0.456, 0.406])
        resnet_std = np.array([0.229, 0.224, 0.225])

        self.resnet_transform = Compose(
            [
                Resize((224, 224)),
                Normalize(mean=resnet_mean.tolist(), std=resnet_std.tolist())
            ]
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

        # in the case of BERT embeddings these are vectors
        token_ids = self._read_text_instance(file_id)
        label = self.id_to_label[file_id]

        return img, token_ids, label

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        images, sequences, labels = zip(*batch)
        images = torch.stack(images)
        sequences = pad_sequence(sequences, batch_first=True)
        return (images, sequences), torch.as_tensor(labels)

    def _read_img_instance(self, file_id):
        img_path = os.path.join(self.data_dir, f'{file_id}.jpg')
        img = torchvision.io.read_image(img_path)
        return img

    def _read_text_instance(self, file_id):
        text_path = os.path.join(self.data_dir, f'{file_id}.txt')
        with open(text_path, 'r', encoding='utf8', errors='replace') as fp:
            tweet = fp.readline()
        tokens = fe_net_tokenize(tweet)
        return torch.as_tensor([self.word_to_id.get(token, self.word_to_id['<UNK>']) for token in tokens])

    @staticmethod
    def _show_img(img):
        np_img = img.numpy().transpose(1, 2, 0)
        pil_img = Image.fromarray(np_img)
        pil_img.show()


class BERTMVSADataset(MVSADataset):
    def _read_text_instance(self, file_id):
        out_file = os.path.join(self.output_dir, f'{file_id}.pkl')
        with open(os.path.join(out_file), 'rb') as fp:
            bert_embeds = pickle.load(fp)
            return bert_embeds
