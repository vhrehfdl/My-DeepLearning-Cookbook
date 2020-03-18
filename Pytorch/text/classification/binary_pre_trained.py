import os

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe


def load_data(train_dir, test_dir):
    NLP = spacy.load('en_core_web_sm')
    tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

    TEXT = data.Field(sequential=True, batch_first=True, lower=True, fix_length=50, tokenize=tokenizer)
    LABEL = data.Field(sequential=False, batch_first=True)

    train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])
    test_data = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])

    train_data, valid_data = train_data.split(split_ratio=0.8)

    return train_data, valid_data, test_data, TEXT, LABEL


def data_preprocissing(train_data, valid_data, test_data, TEXT, LABEL, device, batch_size):
    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    train_iter, val_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=batch_size, device=device,
                                                      sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)
    test_iter = data.Iterator(test_data, batch_size=batch_size, device=device, shuffle=False, sort=False, sort_within_batch=False)

    return train_iter, val_iter, test_iter, TEXT, LABEL


class BasicModel(nn.Module):
    def __init__(self, hidden_dim, n_vocab, embed_dim, n_classes, word_embeddings):
        super(BasicModel, self).__init__()
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.embed.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.linear = nn.Linear(embed_dim * 50, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)   # 2차원 Flatten
        x = self.linear(x)
        print(x.shape)
        logit = self.out(x)

        return logit


class TextCNN(nn.Module):
    def __init__(self, hidden_dim, n_vocab, embed_dim, n_classes, word_embeddings):
        super(TextCNN, self).__init__()
        num_channels = 100
        kernel_size = [3, 4, 5]
        max_sen_len = 50
        dropout_keep = 0.8

        self.embeddings = nn.Embedding(n_vocab, embed_dim)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_sen_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_sen_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim, out_channels=num_channels,
                      kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_sen_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout_keep)

        # Fully-Connected Layer
        self.fc = nn.Linear(num_channels * len(kernel_size), n_classes)

        # Softmax non-linearity
        self.softmax = nn.Softmax()

    def forward(self, x):
        # x.shape = (max_sen_len, batch_size)
        embedded_sent = self.embeddings(x).permute(0, 2, 1)
        # embedded_sent.shape = (batch_size=64,embed_size=300,max_sen_len=20)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)  # shape=(64, num_channels, 1) (squeeze 1)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        final_out = self.fc(final_feature_map)

        return self.softmax(final_out)


def train(model, optimizer, train_iter, device):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter, device):
    model.eval()
    corrects, total_loss = 0, 0
    print(val_iter)
    for batch in val_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


def save_model(best_val_loss, val_loss, model, model_dir):
    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), model_dir)


def main():
    # 하이퍼파라미터
    batch_size = 64
    lr = 0.001
    EPOCHS = 3
    n_classes = 2
    embedding_dim = 300
    hidden_dim = 32

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    base_dir = "../../.."
    train_dir = base_dir + "/Data/binary_train_data.csv"
    test_dir = base_dir + "/Data/binary_test_data.csv"
    model_dir = "./snapshot/txtclassification.pt"

    train_data, valid_data, test_data, TEXT, LABEL = load_data(train_dir, test_dir)
    train_iter, val_iter, test_iter, TEXT, LABEL = data_preprocissing(train_data, valid_data, test_data, TEXT, LABEL, device, batch_size)

    vocab_size = len(TEXT.vocab)
    word_embeddings = TEXT.vocab.vectors

    # model = BasicModel(hidden_dim, vocab_size, embedding_dim, n_classes, word_embeddings).to(device)
    model = TextCNN(hidden_dim, vocab_size, embedding_dim, n_classes, word_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = None
    for e in range(1, EPOCHS + 1):
        train(model, optimizer, train_iter, device)
        val_loss, val_accuracy = evaluate(model, val_iter, device)
        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))
        save_model(best_val_loss, val_loss, model, model_dir)

    model.load_state_dict(torch.load(model_dir))
    test_loss, test_acc = evaluate(model, test_iter, device)
    print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))


if __name__ == '__main__':
    main()