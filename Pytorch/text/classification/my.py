import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from torchtext import data
from torchtext.data import Iterator
from torchtext.data import TabularDataset


# load data from csv file.
def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    return train_x, train_y, test_x, test_y, val_x, val_y


def data_preprocissing(train_x, test_x, val_x):
    NLP = spacy.load('en_core_web_sm')
    tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=30)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train_data = TabularDataset.splits(path='./data/',
                                       train='train_path',
                                       valid='valid_path',
                                       test='test_path',
                                       format='tsv',
                                       fields=[('text', TEXT), ('label', LABEL)])

    train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('text', TEXT), ('label', LABEL)])
    TEXT.build_vocab(train_data, min_freq=1)

    train_loader = Iterator(train_data, batch_size=3, device="cuda", repeat=False)

    for batch in train_loader:
        break

    print(batch.text)
    print(batch.label)

    VOCAB_SIZE = len(train_loader.get_vocab())
    EMBED_DIM = 32
    NUN_CLASS = len(train_loader.get_labels())

    print(VOCAB_SIZE)
    print(NUN_CLASS)


def main():
    base_dir = "../../.."

    train_dir = base_dir + "/Data/binary_train_data.csv"
    test_dir = base_dir + "/Data/binary_test_data.csv"

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)
    train_x, test_x, val_x, tokenizer = data_preprocissing(train_x, test_x, val_x)



if __name__ == '__main__':
    main()
