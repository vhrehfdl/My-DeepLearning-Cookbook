import random
import time

import spacy
import torch
import torch.nn as nn
from torchtext import data
from torchtext.data import TabularDataset

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

NLP = spacy.load('en_core_web_sm')
tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

TEXT = data.Field(tokenize=tokenizer, batch_first=True)
LABEL = data.LabelField(dtype=torch.float, is_target=True)

# train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
base_dir = "../../.."

train_dir = base_dir + "/Data/binary_train_data.csv"
test_dir = base_dir + "/Data/binary_test_data.csv"

train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])
test_data = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])

print(train_data)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

print(TEXT.vocab.freqs.most_common(20))
print(TEXT.vocab.itos[:10])
print(LABEL.vocab.stoi)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
#     (train_data, valid_data, test_data),
#     batch_size=BATCH_SIZE,
#     device=device)

train_iterator = data.Iterator(train_data, batch_size=64, device=device, shuffle=True)
valid_iterator = data.Iterator(valid_data, batch_size=64, device=device, shuffle=True)
test_iterator = data.Iterator(test_data, batch_size=64, device=device, shuffle=True)

print('훈련 데이터의 미니 배치의 개수 : {}'.format(len(train_iterator)))
print('테스트 데이터의 미니 배치의 개수 : {}'.format(len(test_iterator)))
print('검증 데이터의 미니 배치의 개수 : {}'.format(len(valid_iterator)))

for batch in train_iterator:
    break
print(batch.text)
print(batch.label)

for batch in train_iterator:
    print(batch.text)
    print(batch.text[1])
    print(batch.text.shape)
    print(batch.label)
    break

for batch in valid_iterator:
    print(batch.text)
    print(batch.text[1])
    print(batch.text.shape)
    print(batch.label)
    break


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.rnn(embedded)

        # output = [sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class Sentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, n_layers, dropout, method):
        super().__init__()

        self.method = method
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * n_layers, output_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.drop(self.embed(x))

        if (self.method == 'LSTM'):
            out, (h, c) = self.lstm(emb)
            h = self.drop(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))

        if (self.method == 'GRU'):
            out, h = self.gru(emb)
            h = self.drop(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))

        return self.fc(h.squeeze(0))


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
# cifar100_model = Sentiment(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 2, 0.5, "LSTM")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The cifar100_model has {count_parameters(model):,} trainable parameters')

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.001)

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.L1Loss()

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 30

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-cifar100_model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


model.load_state_dict(torch.load('tut1-cifar100_model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')