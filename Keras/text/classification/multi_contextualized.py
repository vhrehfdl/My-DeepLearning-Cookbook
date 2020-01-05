import keras
from sklearn import preprocessing

import keras.backend as K
import keras.layers as layers
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from keras.callbacks import ModelCheckpoint
from keras.engine import Layer
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers import SpatialDropout1D, add, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# load data from csv file.
def load_data(train_dir, test_dir, category_size):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["turn3"], train["label"]
    test_x, test_y = test["turn3"], test["label"]
    val_x, val_y = val["turn3"], val["label"]

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    val_y = encoder.fit_transform(val_y)

    train_y = keras.utils.to_categorical(train_y, category_size)
    val_y = keras.utils.to_categorical(val_y, category_size)

    return train_x, train_y, test_x, test_y, val_x, val_y


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# 첫번째 모델
# ELMo 256 차원 단일 모델
def build_model_elmo(category_size):
    input_layer = layers.Input(shape=(1,), dtype="string")

    embedding = ElmoEmbeddingLayer()(input_layer)
    dense_elmo = layers.Dense(256, activation='relu')(embedding)

    output_layer = Dense(category_size, activation='softmax')(dense_elmo)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model


def evaluate(model, test_x, test_y, target_names):
    predictions = model.predict(test_x)

    y_pred = predictions.argmax(axis=-1)

    print("Accuracy: %.2f%%" % (accuracy_score(test_y, y_pred) * 100.0))
    print(classification_report(test_y, y_pred, target_names=target_names))

    index, count = np.unique(y_pred, return_counts=True)
    print(index)
    print(count)


def create_callbacks(model_dir):
    checkpoint_callback = ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    return [checkpoint_callback]


def main():
    base_dir = "../../.."

    train_dir = base_dir + "/Data/multi_train_data.csv"
    test_dir = base_dir + "/Data/multi_test_data.csv"

    model_dir = base_dir + "/Model"

    category_num = 4
    target_names = ['0', '1', '2', '3']

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir, category_num)
    train_x, val_x, test_x = train_x.tolist(), val_x.tolist(), test_x.tolist()

    train_x = [' '.join(t.split()[0:250]) for t in train_x]
    train_x = np.array(train_x, dtype=object)[:, np.newaxis]

    val_x = [' '.join(t.split()[0:250]) for t in val_x]
    val_x = np.array(val_x, dtype=object)[:, np.newaxis]

    test_x = [' '.join(t.split()[0:250]) for t in test_x]
    test_x = np.array(test_x, dtype=object)[:, np.newaxis]

    model = build_model_elmo(category_num)

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=10, batch_size=128, validation_data=(val_x, val_y), callbacks=callbacks)

    evaluate(model, test_x, test_y, target_names)


if __name__ == '__main__':
    main()
