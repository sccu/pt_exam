import torch
import torch.nn as nn

from nce import NCELoss


class Word2Vec(nn.Module):
  def __init__(self, vocab_size, emb_size):
    super(Word2Vec, self).__init__()
    self.embedding = nn.Embedding(vocab_size, emb_size)
    self.out_proj = nn.Linear(emb_size, vocab_size)

    if torch.cuda.is_available():
      print("Cuda available.")
      self.embedding = self.embedding.cuda()
      self.out_proj = self.out_proj

  def forward(self, x):
    l = self.embedding(x)
    out = self.out_proj(l)
    return out

#optim = nn.NLLLoss
#criterion = nn.MSELoss()
optim = NCELoss()
model = Word2Vec()
