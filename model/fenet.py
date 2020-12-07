import json

import torch
from hydra.utils import to_absolute_path
from torch import hub
from torch.nn import Module, Sequential, Linear, Tanh, Softmax, Conv1d, Sigmoid, Dropout

from model.glove import GloVeEmbedding


class FENet(Module):
    # TODO docstring
    def __init__(self, vocab_filepath, h_dim, n_kernels, kernel_size, dropout_rate):
        super().__init__()
        self.text_enc = TextEncoding(vocab_filepath)
        self.img_enc = ImageEncoding('resnet152')

        img_dim = 49
        text_dim = 300
        self.text_img_fuse = InformationFusion(target_dim=text_dim,
                                               auxiliary_dim=img_dim,
                                               h_dim=h_dim,
                                               dropout_rate=dropout_rate)
        self.img_text_fuse = InformationFusion(target_dim=img_dim,
                                               auxiliary_dim=text_dim,
                                               h_dim=h_dim,
                                               dropout_rate=dropout_rate)

        self.img_extract = InformationExtraction(img_dim,
                                                 n_kernels,
                                                 kernel_size,
                                                 dropout_rate)
        self.text_extract = InformationExtraction(text_dim,
                                                  n_kernels,
                                                  kernel_size,
                                                  dropout_rate)

        self.lin_out = Linear(2 * n_kernels, 3)

    def forward(self, inputs):
        images, tokens = inputs

        x_txt = self.text_enc(tokens)
        x_img = self.img_enc(images)

        fused_txt = self.text_img_fuse([x_txt, x_img])
        fused_img = self.img_text_fuse([x_img, x_txt])

        x_txt = self.text_extract(fused_txt)
        x_img = self.img_extract(fused_img)

        x = torch.cat([x_txt, x_img], dim=-1)
        x = self.lin_out(x)
        return x


class TextEncoding(Module):
    """Text encoding layer of the FENet
    """

    def __init__(self, vocab_filepath):
        """
        :param vocab_filepath: path to json file that contains a token to id mapping.
        """
        super().__init__()

        with open(to_absolute_path(vocab_filepath), 'r') as f:
            word_to_id = json.load(f)

        self.glove = GloVeEmbedding(word_to_id)

    def forward(self, token_ids):
        return self.glove(token_ids)


class ImageEncoding(Module):
    """Image encoding layer of the FENet
    """

    def __init__(self, resnet_version='resnet152', repo_or_dir='pytorch/vision:v0.6.0'):
        """
        :param resnet_version: which resnet to use: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        :param repo_or_dir:
        """
        super().__init__()
        # only use layers until last feature map extraction
        resnet = hub.load(repo_or_dir, resnet_version, pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = Sequential(*modules)

    def forward(self, images):
        x = self.resnet(images)
        x = x.flatten(start_dim=2)
        return x


class InformationFusion(Module):
    def __init__(self, target_dim, auxiliary_dim, h_dim, dropout_rate):
        super().__init__()

        self.t_project = Linear(target_dim, h_dim)
        self.a_project = Linear(auxiliary_dim, h_dim)
        self.fuse_project = Linear(target_dim + auxiliary_dim, target_dim)

        self.dp = Dropout(dropout_rate)
        self.tanh = Tanh()
        self.row_softmax = Softmax(dim=1)

    def forward(self, inputs):
        target, auxiliary = inputs
        target, auxiliary = self.dp(target), self.dp(auxiliary)

        # project into shared space
        target_h = self.tanh(self.t_project(target))
        auxiliary_h = self.tanh(self.a_project(auxiliary)).transpose(2, 1)

        # compute attention matrix and weight
        attention_mat = self.row_softmax(target_h @ auxiliary_h)
        # weighted sum of aux vectors for each target vector
        weighted_aux = attention_mat @ auxiliary

        x = torch.cat([target, weighted_aux], dim=-1)
        x = self.dp(x)
        x = self.tanh(self.fuse_project(x))
        return x.transpose(2, 1)


class InformationExtraction(Module):
    def __init__(self, in_channels, n_kernels, kernel_size, dropout_rate):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_extract = Conv1d(in_channels, n_kernels, kernel_size, padding=padding)
        self.conv_gate = Conv1d(in_channels, n_kernels, kernel_size, padding=padding)

        self.dp = Dropout(dropout_rate)
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        inputs = self.dp(inputs)
        features = self.sigmoid(self.conv_extract(inputs))
        gates = self.tanh(self.conv_gate(inputs))

        gated = features * gates
        pooled, _ = gated.max(dim=-1)

        return pooled
