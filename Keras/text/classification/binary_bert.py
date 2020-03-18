import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

sess = tf.Session()
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


# gpu setting.
def set_env():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    session


# load data from csv file.
def load_data(train_dir, test_dir, max_seq_length):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x = train['text'].tolist()
    train_x = [' '.join(t.split()[0:max_seq_length]) for t in train_x]
    train_x = np.array(train_x, dtype=object)[:, np.newaxis]
    train_y = train['label'].tolist()

    val_x = val['text'].tolist()
    val_x = [' '.join(t.split()[0:max_seq_length]) for t in val_x]
    val_x = np.array(val_x, dtype=object)[:, np.newaxis]
    val_y = val['label'].tolist()

    test_x = test['text'].tolist()
    test_x = [' '.join(t.split()[0:max_seq_length]) for t in test_x]
    test_x = np.array(test_x, dtype=object)[:, np.newaxis]
    test_y = test['label'].tolist()

    return train_x, train_y, test_x, test_y, val_x, val_y


class PaddingInputExample(object):
    """"""


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module():
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(tokenizer, example, max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )


def convert_text_to_examples(texts, labels):
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples


class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables

        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "pooled_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# Build model
def build_model_bert(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


def evaluate(model, test_input_ids, test_input_masks, test_segment_ids, test_labels):
    post_save_preds = model.predict([test_input_ids, test_input_masks, test_segment_ids])
    y_pred = (post_save_preds > 0.5)

    accuracy = accuracy_score(test_labels, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(test_labels, y_pred, target_names=["0", "1"]))


def main():
    ### Parameter
    max_length = 50

    ### Directory Setting.
    base_dir = "../../.."

    train_dir = base_dir + "/Data/binary_train_data.csv"
    test_dir = base_dir + "/Data/binary_test_data.csv"

    model_dir = base_dir + "/Model"


    ### Flow
    set_env()

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir, max_length)

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module()

    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_x, train_y)
    val_examples = convert_text_to_examples(val_x, val_y)
    test_examples = convert_text_to_examples(test_x, test_y)

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels) = convert_examples_to_features(tokenizer, train_examples, max_length)
    (val_input_ids, val_input_masks, val_segment_ids, val_labels) = convert_examples_to_features(tokenizer, val_examples, max_length)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels) = convert_examples_to_features(tokenizer, test_examples, max_length)

    print(train_labels)

    model = build_model_bert(max_length)
    initialize_vars(sess)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', save_best_only=True, verbose=1)

    model.fit(
        [train_input_ids, train_input_masks, train_segment_ids], train_labels,
        validation_data=([val_input_ids, val_input_masks, val_segment_ids], val_labels),
        epochs=2,
        batch_size=512,
        callbacks=[cp_callback]
    )

    evaluate(model, test_input_ids, test_input_masks, test_segment_ids, test_labels)


if __name__ == '__main__':
    main()
