import torch
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def restrict_len(txt, min_len, max_len=50):
    new_egs = []
    for ex in txt.examples:
        if min_len >= len(ex.premise) or len(ex.premise) >= max_len:
            new_egs.append(ex)
    return new_egs


batch_size = 128
# Load SNLI dataset
TEXT = torchtext.data.Field(lower=True, include_lengths=True)
LABEL = torchtext.data.Field()
train_txt, val_txt, test_txt = torchtext.datasets.SNLI.splits(TEXT, LABEL)
TEXT.build_vocab(train_txt)
LABEL.build_vocab(train_txt)
# TEXT.numericalize(train_txt.example)
train_txt.examples = restrict_len(train_txt, 30)
val_txt.examples = restrict_len(val_txt, 30)
test_txt.examples = restrict_len(test_txt, 30)
train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_txt, val_txt, test_txt), batch_size=batch_size, device=device)



