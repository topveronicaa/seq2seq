import torchtext
import torch
import torch.nn as nn
import math
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SNLI dataset
TEXT = torchtext.data.Field(lower=True)
LABEL = torchtext.data.Field()
train_txt, val_txt, test_txt = torchtext.datasets.SNLI.splits(TEXT, LABEL)
TEXT.build_vocab(train_txt)
LABEL.build_vocab(train_txt)
train_iter, dev_iter, test_iter = torchtext.data.BucketIterator.splits(
            (train_txt, val_txt, test_txt), batch_size=batch_size, device=device)

ninp = 128  # number of features
nhid = 200  # hidden space dimension
ntoken = len(TEXT.vocab.stoi)
zero_tensor = torch.zeros(batch_size, dtype=torch.long, device=device)  # batch_size


def KL_estimator(z):
    # z [batch_size, nhid]
    N = z.shape[0] - 1
    d = z.shape[1]
    # calculate pairwise distance
    D = torch.cdist(z, z)
    # mask the diagonal value
    mask = torch.eye(len(z), len(z)).bool().to(device)
    D.masked_fill_(mask, float('inf'))
    # get R
    R, _ = D.min(dim=0)
    R += 0.01
    Y = math.log(N) + d * torch.log(R)
    estimate = Y.mean().item()
    return estimate


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ntoken: size of the embedding dictionary; ninp: size of each embedding vector
        self.embedding = nn.Embedding(ntoken, ninp)
        # ninp – The number of expected features in the input x; nhid – The number of features in the hidden state h
        self.rnn = nn.GRU(ninp, nhid)
        self.batch_norm = nn.BatchNorm1d(num_features=nhid, affine=False)

    def forward(self, src):
        # src [X, batch_size]
        embedded = self.embedding(src)
        # embedded [X, batch_size, ninp]
        outputs, hidden = self.rnn(embedded)
        # batch norm
        hidden = torch.squeeze(hidden, 0)
        hidden = self.batch_norm(hidden)
        return hidden  # [batch_size, nhid]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(ntoken, ninp)
        self.rnn = nn.GRUCell(ninp + nhid, nhid)
        self.linear = nn.Linear(nhid, ntoken)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, context):
        # input [batch_size]
        # context [batch_size, nhid]
        emb_con = self.embedding(input)
        # emb_con [batch_size, ninp]
        emb_con = torch.cat((emb_con, context), 1)
        # emb_con [batch_size, ninp + nhid]
        hidden = self.rnn(emb_con, hidden)
        # hidden [batch_size, nhid]
        output = self.linear(hidden)
        # output [batch_size, ntoken]
        output = self.softmax(output)
        # output [batch_size, ntoken]
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, batch):
        context = torch.squeeze(self.encoder(batch), 0)
        hidden = context
        trg_len = batch.shape[0]
        cur_bs = batch.shape[1]  # current batch size
        output = torch.zeros(cur_bs, dtype=torch.long, device=device)  # batch_size
        outputs = torch.zeros(trg_len, cur_bs, ntoken).to(self.device)
        for t in range(0, trg_len):
            # calculate loss
            output, hidden = decoder(output, hidden, context)
            # save class prob
            outputs[t] = output
            # determine next input
            predicted = torch.argmax(output, dim=1)
            output = predicted
        return outputs, context


def evaluate(seq2seq, criterion, iter):
    # seq2seq.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(iter):
            output, context = seq2seq(batch.premise)
            # reshape to [trg_len * batch size, ntoken]
            output = output.view(-1, output.shape[-1])
            target = batch.premise.view(-1)
            loss = criterion(output, target) - 0.01 * KL_estimator(context)
            total_loss += loss.item()
    return total_loss / batch_idx


encoder = Encoder().to(device)
decoder = Decoder().to(device)
seq2seq = Seq2Seq(encoder, decoder, device).to(device)
criterion = nn.NLLLoss()

seq2seq.load_state_dict(torch.load('best_model_bn.pt', map_location=torch.device('cpu')))
test_loss = evaluate(seq2seq, criterion, test_iter)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

# test reconstruction, bleu score
dict = TEXT.vocab.itos
seq2seq.eval()
bleu_scores = []
preds = []
trgs = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_iter):
        output, context = seq2seq(batch.premise)
        # reshape to [trg_len * batch size, ntoken]
        target = batch.premise
        predicted = torch.argmax(output, dim=2)
        num_sentence = target.shape[1]
        num_word = predicted.shape[0]
        for i in range(num_sentence):
            s_predicted = predicted[:, i]
            s_trg = target[:, i]
            s1 = []
            s2 = []
            for w in range(num_word):
                s1.append(dict[s_predicted[w]])
                s2.append(dict[s_trg[w]])
            preds.append(s1)
            trgs.append(s2)
            bleu_scores.append(sentence_bleu([s2], s1))
    avg_bleu = np.array(bleu_scores).mean()  # 0.48029889658941854

# dict = TEXT.vocab.itos
# seq2seq.eval()
# bleu_scores = []
# preds = []
# trgs = []
# with torch.no_grad():
#     for batch_idx, batch in enumerate(test_iter):
#         output, context = seq2seq(batch.premise)
#         # reshape to [trg_len * batch size, ntoken]
#         target = batch.premise
#         predicted = torch.argmax(output, dim=2)
#         num_sentence = target.shape[1]
#         num_word = predicted.shape[0]
#         for i in range(num_sentence):
#             s_predicted = predicted[:, i]
#             s_trg = target[:, i]
#             s1 = []
#             s2 = []
#             for w in range(num_word):
#                 s1.append(dict[s_predicted[w]])
#                 s2.append(dict[s_trg[w]])
#             preds.append(s1)
#             trgs.append(s2)
#             bleu_scores.append(sentence_bleu([s2], s1))
#     avg_bleu = np.array(bleu_scores).mean()

# sampling
# with torch.no_grad():
#     context = torch.randn(batch_size, nhid)
#     hidden = context
#     trg_len = 10
#     cur_bs = batch_size
#     output = torch.zeros(cur_bs, dtype=torch.long, device=device)  # batch_size
#     outputs = torch.zeros(trg_len, cur_bs, ntoken).to(device)
#     for t in range(0, trg_len):
#         # calculate loss
#         output, hidden = decoder(output, hidden, context)
#         # save class prob
#         outputs[t] = output
#         # determine next input
#         predicted = torch.argmax(output, dim=1)
#         output = predicted
#     predicted = torch.argmax(outputs, dim=2)
#     num_sentence = batch_size
#     sentences = []
#     for i in range(num_sentence):
#         s_predicted = predicted[:, i]
#         s1 = []
#         for w in range(trg_len):
#             s1.append(dict[s_predicted[w]])
#         sentences.append(s1)

