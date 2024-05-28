
from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer

tokenizer = Tokenizer()
loader = DataLoader(
    ext=(".en", ".de"),
    tokenize_en=tokenizer.tokenize_en,
    tokenize_de=tokenizer.tokenize_de,
    init_token="<sos>",
    eos_token="<eos>",
)

train_, valid, test = loader.make_dataset()
loader.build_vocab(train_data=train_, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(
    train_, valid, test, batch_size=batch_size, device=device
)

src_pad_idx = loader.source.vocab.stoi["<pad>"]
trg_pad_idx = loader.target.vocab.stoi["<pad>"]
trg_sos_idx = loader.target.vocab.stoi["<sos>"]

enc_voc_size = len(loader.source.vocab)
dec_voc_size = len(loader.target.vocab)
