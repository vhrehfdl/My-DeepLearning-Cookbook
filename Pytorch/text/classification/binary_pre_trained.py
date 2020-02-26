import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe


# load data from csv file.
def load_data(train_dir, test_dir):
    tokenizer = lambda x: x.split()

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True, fix_length=200)
    LABEL = data.LabelField(dtype=torch.float)

    train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])
    test_data = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])

    train_data, valid_data = train_data.split(random_state=random.seed(1234))

    return train_data, valid_data, test_data, TEXT, LABEL


def data_preprocissing(train_data, valid_data, test_data, TEXT, LABEL, device):
    TEXT.build_vocab(train_data, vectors=GloVe(name='840B', dim=300))
    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)

    print("vocab size => ", vocab_size)

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=32,
                                                                   sort_key=lambda x: len(x.text), repeat=False,
                                                                   shuffle=True, device=device)

    return train_iterator, valid_iterator, test_iterator, TEXT, LABEL, word_embeddings, vocab_size


class BasicModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(BasicModel, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
        self.linear = nn.Linear(vocab_size, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)


        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        self.linear()
        final_output = self.label(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)

        return final_output


class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(LSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)  # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence)  # embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = input.permute(1, 0, 2)  # input.size() = (num_sequences, batch_size, embedding_length)

        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
        # print("check =>", input.shape)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        print("check1 =>", final_hidden_state.shape)
        final_output = self.label(final_hidden_state[-1])  # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        print("check2 =>", final_hidden_state[-1].shape)
        print("check3 =>", final_output.shape)

        return final_output


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
        if batch.text.shape[0] == 32:
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
            if batch.text.shape[0] == 32:
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
    train_iterator, valid_iterator, test_iterator, TEXT, LABEL, word_embeddings, vocab_size = data_preprocissing(train_data, val_data, test_data, TEXT, LABEL, device)

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    # model = BasicModel(128, OUTPUT_DIM, HIDDEN_DIM, vocab_size, EMBEDDING_DIM, word_embeddings)
    model = LSTMClassifier(128, OUTPUT_DIM, HIDDEN_DIM, vocab_size, EMBEDDING_DIM, word_embeddings)
    # model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM)

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