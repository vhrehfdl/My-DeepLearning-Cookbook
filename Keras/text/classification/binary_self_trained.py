import os

import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import SpatialDropout1D, add, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# gpu setting.
def set_env():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    session


# load data from csv file.
def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    return train_x, train_y, test_x, test_y, val_x, val_y


# convert text data to vector.
def data_preprocissing(train_x, test_x, val_x):
    CHARS_TO_REMOVE = r'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

    train_x = train_x.tolist()
    test_x = test_x.tolist()
    val_x = val_x.tolist()

    tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
    tokenizer.fit_on_texts(train_x + test_x + val_x)  # Make dictionary

    # Text match to dictionary.
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    val_x = tokenizer.texts_to_sequences(val_x)

    temp_list = []
    total_list = list(train_x) + list(test_x) + list(val_x)

    for i in range(0, len(total_list)):
        temp_list.append(len(total_list[i]))

    max_len = max(temp_list)

    train_x = sequence.pad_sequences(train_x, maxlen=max_len, padding='post')
    test_x = sequence.pad_sequences(test_x, maxlen=max_len, padding='post')
    val_x = sequence.pad_sequences(val_x, maxlen=max_len, padding='post')

    return train_x, test_x, val_x, tokenizer


def build_model_basic(size, vocab_size):
    ### Hyper Parameter
    embedding_dim = 300
    hidden_units = 64

    ### Model Architecture
    input_layer = Input(shape=(size,))

    embedding_layer = Embedding(vocab_size, embedding_dim)(input_layer)

    dense_layer = Flatten()(embedding_layer)
    hidden_layer = Dense(hidden_units, activation='sigmoid')(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_model_lstm(size, vocab_size):
    ### Hyper Parameter
    embedding_dim = 300
    lstm_units = 128
    hidden_units = 512

    ### Model Architecture
    input_layer = Input(shape=(size,))

    embedding_layer = Embedding(vocab_size, embedding_dim)(input_layer)

    lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_layer)
    lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_layer)

    hidden_layer = concatenate([GlobalMaxPooling1D()(lstm_layer), GlobalAveragePooling1D()(lstm_layer)])
    hidden_layer = add([hidden_layer, Dense(hidden_units, activation='relu')(hidden_layer)])
    hidden_layer = add([hidden_layer, Dense(hidden_units, activation='relu')(hidden_layer)])

    output_layer = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# TextCNN
def build_model_cnn(size, vocab_size):
    ### Hyper Parameter
    embedding_dim = 300
    num_filters = 128
    filter_sizes = [3, 4, 5]

    ### Model Architecture
    input_layer = Input(shape=(size,))
    embedding_layer = Embedding(vocab_size, embedding_dim)(input_layer)

    pooled_outputs = []
    for filter_size in filter_sizes:
        x = Conv1D(num_filters, filter_size, activation='relu')(embedding_layer)
        x = MaxPool1D(pool_size=2)(x)
        pooled_outputs.append(x)

    merged = concatenate(pooled_outputs, axis=1)
    dense_layer = Flatten()(merged)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def evaluate(model, test_x, test_y):
    prediction = model.predict(test_x)
    y_pred = (prediction > 0.5)

    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(test_y, y_pred, target_names=["0", "1"]))


def create_callbacks(model_dir):
    checkpoint_callback = ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    return [checkpoint_callback]


def main():
    ### Directory Setting.
    base_dir = "../../.."

    train_dir = base_dir + "/Data/binary_train_data.csv"
    test_dir = base_dir + "/Data/binary_test_data.csv"

    model_dir = base_dir + "/Model"


    ### Flow
    set_env()

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)

    train_x, test_x, val_x, tokenizer = data_preprocissing(train_x, test_x, val_x)

    vocab_size = len(tokenizer.word_index)

    # model = build_model_basic(train_x.shape[1], vocab_size)
    # model = build_model_lstm(train_x.shape[1], vocab_size)
    model = build_model_cnn(train_x.shape[1], vocab_size)

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=3, batch_size=64, validation_data=(val_x, val_y), callbacks=callbacks)

    evaluate(model, test_x, test_y)


if __name__ == '__main__':
    main()
