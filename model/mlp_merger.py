import torch
from torch.nn import Module, Sequential, Linear, Dropout, ReLU

from model.fenet import TextEncoding, ImageEncoding


class MLPMerger(Module):
    """Simple baseline model that uses an MLP to merge mean of text and img features.
    """

    def __init__(self, vocab_filepath, h_dim, n_hidden_layers, dropout_rate, img_backbone):
        super().__init__()
        self.text_enc = TextEncoding(vocab_filepath)
        self.img_enc = ImageEncoding(img_backbone)

        self.text_project = Linear(300, h_dim)
        self.img_project = Linear(49, h_dim)

        self.dp = Dropout(dropout_rate)

        self.merge_mlp = self._get_mlp(h_dim, n_hidden_layers, dropout_rate)

    def forward(self, inputs):
        images, tokens = inputs

        x_txt = self.text_enc(tokens)
        x_img = self.img_enc(images)

        x_txt = self.dp(x_txt.mean(dim=1))
        x_img = self.dp(x_img.mean(dim=1))

        x_txt = self.text_project(x_txt)
        x_img = self.img_project(x_img)

        x = torch.cat([x_txt, x_img], dim=-1)
        x = self.dp(x)
        return self.merge_mlp(x)

    @staticmethod
    def _get_mlp(h_dim, n_hidden_layers, dropout_rate):
        modules = [Linear(2 * h_dim, h_dim), ReLU()]

        for _ in range(n_hidden_layers):
            modules.append(Dropout(dropout_rate))
            modules.append(Linear(h_dim, h_dim))
            modules.append(ReLU())

        modules.append(Linear(h_dim, 3))

        return Sequential(*modules)
