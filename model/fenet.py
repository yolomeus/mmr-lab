import json

import torch
from hydra.utils import to_absolute_path
from torch import hub
from torch.nn import Module, Sequential, Linear, Tanh, Softmax, Conv1d, Sigmoid, Dropout, Identity

from model.glove import GloVeEmbedding


class FENet(Module):
    """Fusion-Extraction Network as described in https://link.springer.com/chapter/10.1007/978-3-030-47436-2_59"""

    def __init__(self,
                 vocab_filepath,
                 h_dim,
                 n_kernels,
                 kernel_size,
                 dropout_rate,
                 text_backbone,
                 img_backbone='resnet152'):
        """

        :param vocab_filepath: path to a json which contains a mapping from tokens to ids.
        :param h_dim: hidden dimension.
        :param n_kernels: number of kernels in the conv layers.
        :param kernel_size: kernel size for the conv layers.
        :param dropout_rate: dropout rate applied before each fully connected and conv layer.
        :param img_backbone: torch hub module to use for image extraction. Currently assumes a resnet.
        """
        super().__init__()

        if text_backbone == 'glove':
            self.text_enc = TextEncoding(vocab_filepath)
            text_dim = 300
        elif text_backbone == 'bert':
            # we already receive vectors for BERT
            self.text_enc = Identity()
            text_dim = 768
        else:
            raise NotImplementedError()

        self.img_enc = ImageEncoding(img_backbone)

        img_dim = 49
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
    """Text encoding layer of the FENet.
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

    def __init__(self, resnet_version='resnet152', repo_or_dir='pytorch/vision:v0.6.0', freeze=True):
        """
        :param resnet_version: which resnet to use: ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        :param repo_or_dir: torch hub repo or model directory.
        """
        super().__init__()
        # only use layers until last feature map extraction
        resnet = hub.load(repo_or_dir, resnet_version, pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = Sequential(*modules)

        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def forward(self, images):
        x = self.resnet(images)
        x = x.flatten(start_dim=2)
        return x


class InformationFusion(Module):
    """Interactive Information Fusion Layer as described in the paper. Takes a target and auxiliary sequence and
    produces a new representation of the target sequence by attending to the auxiliary sequence.
    """

    def __init__(self, target_dim, auxiliary_dim, h_dim, dropout_rate):
        """

        :param target_dim: hidden dimension of attending sequence.
        :param auxiliary_dim: hidden dimension of the sequence that is attended to.
        :param h_dim: dimension of the common space, both sequences are projected to.
        :param dropout_rate: Dropout before projections.
        """
        super().__init__()

        self.t_project = Linear(target_dim, h_dim)
        self.a_project = Linear(auxiliary_dim, h_dim)
        self.fuse_project = Linear(target_dim + auxiliary_dim, target_dim)

        self.dp = Dropout(dropout_rate)
        self.tanh = Tanh()
        self.row_softmax = Softmax(dim=-1)

    def forward(self, inputs, attention_mask=None):
        """

        :param inputs: tuple of target and auxiliary sequence.
        :param attention_mask: mask for the attention matrix that masks out all pad token positions, with dimensions
        [batch_size x target_seq_len x aux_seq_len].
        """
        target, auxiliary = inputs
        target, auxiliary = self.dp(target), self.dp(auxiliary)

        # project into shared space
        target_h = self.tanh(self.t_project(target))
        auxiliary_h = self.tanh(self.a_project(auxiliary)).transpose(2, 1)

        raw_attention_mat = target_h @ auxiliary_h
        # don't attend over padding tokens
        if attention_mask is not None:
            raw_attention_mat += attention_mask
        # compute attention matrix and weight
        attention_mat = self.row_softmax(raw_attention_mat)
        # weighted sum of aux vectors for each target vector
        weighted_aux = attention_mat @ auxiliary

        x = torch.cat([target, weighted_aux], dim=-1)
        x = self.dp(x)
        x = self.tanh(self.fuse_project(x))
        return x.transpose(2, 1)


class InformationExtraction(Module):
    """Specific Information Extraction Layer as in the paper. Inputs are convolved with two sets of kernels, then one of
    the outputs is used to gate the other one and max pooling is applied.
    """

    def __init__(self, in_channels, n_kernels, kernel_size, dropout_rate):
        """

        :param in_channels: hidden dimension of input sequence.
        :param n_kernels: number of conv kernels/new hidden dimension.
        :param kernel_size: size of each kernel.
        :param dropout_rate:
        """
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
