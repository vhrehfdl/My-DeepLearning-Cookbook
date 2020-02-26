import pandas as pd
import torch
import torch.optim as optim
from config import Config
from model import TextCNN
from sklearn.model_selection import train_test_split
from torch import nn
from utils import Dataset, evaluate_model


if __name__ == '__main__':
    base_dir = "../../../.."

    train_dir = base_dir + "/Data/binary_train_data.csv"
    test_dir = base_dir + "/Data/binary_test_data.csv"

    model_dir = base_dir + "/Model"
    embedding_dir = base_dir + "/Data/glove.840B.300d.txt"

    config = Config()
    dataset = Dataset(config)
    dataset.load_data(embedding_dir, train_dir, test_dir)

    print(dataset.vocab)

    # Create Model with specified optimizer and loss function
    ##############################################################
    model = TextCNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################

    train_losses = []
    val_accuracies = []

    for i in range(config.max_epochs):
        print("Epoch: {}".format(i))
        train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Test Accuracy: {:.4f}'.format(test_acc))