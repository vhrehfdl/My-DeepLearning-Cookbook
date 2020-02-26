import pandas as pd
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


# load data from csv file.
def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir, delimiter="\t")
    test = pd.read_csv(test_dir, delimiter="\t")

    train = train.dropna()
    test = test.dropna()

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["document"], train["label"]
    test_x, test_y = test["document"], test["label"]
    val_x, val_y = val["document"], val["label"]

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


# BI_LSTM
def build_model_lstm(size, vocab_size):
    LSTM_UNITS = 128
    DENSE_HIDDEN_UNITS = 512

    input_layer = Input(shape=(size,))
    embedding_layer = Embedding(vocab_size, 300)(input_layer)
    embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(embedding_layer)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])

    output_layer = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# TextCNN
def build_model_cnn(size, vocab_size):
    num_filters = 128
    filter_sizes = [3, 4, 5]

    input_layer = Input(shape=(size,))
    embedding_layer = Embedding(vocab_size, 300)(input_layer)
    embedding_layer = SpatialDropout1D(0.2)(embedding_layer)

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
    checkpoint_callback = ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5",
                                          monitor='val_acc', verbose=1, save_best_only=True)
    return [checkpoint_callback]


def main():
    base_dir = "../../.."

    # train_dir = base_dir + "/Data/binary_train_data.csv"
    # test_dir = base_dir + "/Data/binary_test_data.csv"

    train_dir = base_dir + "/Data/ratings_train.txt"
    test_dir = base_dir + "/Data/ratings_test.txt"

    model_dir = base_dir + "/Model"

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)
    train_x, test_x, val_x, tokenizer = data_preprocissing(train_x, test_x, val_x)

    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    print(vocab_size)

    print(train_x.shape)
    print(train_x.shape[0])
    print(train_x.shape[1])

    # model = build_model_basic(train_x.shape[1], embedding_matrix)
    # model = build_model_lstm(train_x.shape[1], embedding_matrix)
    model = build_model_cnn(train_x.shape[1], vocab_size)

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=10, batch_size=128, validation_data=(val_x, val_y), callbacks=callbacks, verbose=2)

    evaluate(model, test_x, test_y)


if __name__ == '__main__':
    main()
