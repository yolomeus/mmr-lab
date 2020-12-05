import logging

from hydra.utils import to_absolute_path
from torch.nn import Module, Embedding, init
from torchtext.vocab import GloVe

logger = logging.getLogger(__name__)


class GloVeEmbedding(Module):
    def __init__(self,
                 word_to_id: dict,
                 dim: int = 300,
                 glove_name: str = '840B',
                 cache: str = 'data/.glove_cache'):
        super().__init__()

        cache = to_absolute_path(cache)
        self.glove_vectors = GloVe(glove_name, dim, cache=cache, unk_init=self._unk_init)
        vocab_size = len(word_to_id)
        self.embeddings = Embedding(vocab_size, dim, padding_idx=0)
        self.word_to_id = word_to_id

        self.unk_count = 0
        self._init_glove_matrix()

    def forward(self, inputs):
        return self.embeddings(inputs)

    def _init_glove_matrix(self):
        i = 0
        for word, idx in self.word_to_id.items():
            word_vec = self.glove_vectors.get_vecs_by_tokens(word)
            self.embeddings.weight.data[idx] = word_vec
            i += 1

        logger.info(f'{i - self.unk_count}/{i} embeddings within GloVe vocab were loaded.')

    def _unk_init(self, tensor):
        self.unk_count += 1
        return init.normal_(tensor)
