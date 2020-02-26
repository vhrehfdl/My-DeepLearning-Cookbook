import random
import time

import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.data import TabularDataset


# load data from csv file.
def load_data(train_dir, test_dir):
    NLP = spacy.load('en_core_web_sm')
    tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

    TEXT = data.Field(tokenize=tokenizer)
    LABEL = data.LabelField(dtype=torch.float)

    train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])
    test_data = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])

    train_data, valid_data = train_data.split(random_state=random.seed(1234))

    return train_data, valid_data, test_data, TEXT, LABEL


def data_preprocissing(train_data, valid_data, test_data, TEXT, LABEL, device):
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)

    train_iterator, valid_iterator = data.BucketIterator.splits((train_data, valid_data), batch_size=64, device=device, shuffle=True, sort_key=lambda x: len(x.text), sort_within_batch=True)
    test_iterator = data.Iterator(test_data, batch_size=64, device=device, shuffle=False)

    # train_iterator, valid_iterator, test_iterator = data.Iterator.splits(
    #     (train_data, valid_data, test_data), device=device, sort_key=lambda x: len(x.text), sort_within_batch=True, shuffle=True, batch_sizes=(64, 64, 64))


    # cnt = 0
    # for batch in train_iterator:
    #     # print(batch.text)
    #     # print(batch.text[1])
    #     print(batch.text.shape)
    #     # print(batch.label)
    #     cnt += 1
    #     if cnt == 3:
    #         break
    #
    # for batch in valid_iterator:
    #     # print(batch.text)
    #     # print(batch.text[1])
    #     print(batch.text.shape)
    #     # print(batch.label)
    #     break

    return train_iterator, valid_iterator, test_iterator, TEXT, LABEL


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # print("text shape => ", text.shape)
        embedded = self.embedding(text)
        # print("embedding shape => ", embedded.shape)
        # print("embedding 실제 값---")
        # print(embedded[0])
        output, hidden = self.rnn(embedded)

        assert torch.equal(output[-1, :, :], hidden.squeeze(0))

        return self.fc(hidden.squeeze(0))


class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
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


def main():
    base_dir = "../../.."

    train_dir = base_dir + "/Data/binary_train_data.csv"
    test_dir = base_dir + "/Data/binary_test_data.csv"

    torch.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, val_data, test_data, TEXT, LABEL = load_data(train_dir, test_dir)
    train_iterator, valid_iterator, test_iterator, TEXT, LABEL = data_preprocissing(train_data, val_data, test_data, TEXT, LABEL, device)

    # print("---"*100)
    # print(vars(train_iterator.dataset.examples[0]))
    # print(valid_iterator.dataset)
    # print(test_iterator.dataset)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    # model = TextSentiment(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.SGD(model.parameters(), lr=0.03)
    criterion = nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)

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
            torch.save(model.state_dict(), 'tut1_model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('tut1_model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


if __name__ == '__main__':
    main()