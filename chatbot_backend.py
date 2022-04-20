import re
import tensorflow as tf

def preprocessing(line):
    line = re.sub(r'[^a-zA-z.?!\']', ' ', line)
    line = re.sub(r'[ ]+', ' ', line)
    return line

def tokenize_data(input_list):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
    tokenizer.fit_on_texts(input_list)
    input_seq = tokenizer.texts_to_sequences(input_list)
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, padding='pre')
    return tokenizer, input_seq

def create_categorical_target(targets):
    word = {}
    categorical_target = []
    counter = 0
    for trg in targets:
        if trg not in word:
            word[trg] = counter
            counter += 1
        categorical_target.append(word[trg])

    categorical_tensor = tf.keras.utils.to_categorical(categorical_target, num_classes=len(word), dtype='int32')
    return categorical_tensor, dict((v, k) for k, v in word.items())

def train_model(tokenizer, input_tensor, target_tensor):
    epochs = 100
    vocab_size = len(tokenizer.word_index) + 1
    embed_dim = 512
    units = 128
    target_length = target_tensor.shape[1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, embed_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, dropout=0.2)),
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(target_length, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(lr=1e-2)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)

    model.fit(input_tensor, target_tensor, epochs=epochs, callbacks=[early_stop])

    return model
