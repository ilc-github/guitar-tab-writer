'''
Testing auto-save model checkpoints (only saves model weights)
Keras docs lstm_text_generation example code.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os


if len(sys.argv) < 2:
    sys.exit('Usage: %s flat_tab_input.txt' % sys.argv[0])

if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: flat_tab_input.txt %s was not found!' % sys.argv[1])

text_filepath = sys.argv[1]



'''
Using base LSTM text generation example from Keras repo
'''

path = text_filepath
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# model.compile(loss='categorical_crossentropy', optimizer='adam')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

checkpoint = ModelCheckpoint( \
                filepath='weights-{epoch:02d}-{loss:.2f}.hdf5', \
                monitor='loss', verbose=0, save_best_only=True, \
                save_weights_only=False, mode='auto')

# checkpoint = ModelCheckpoint( \
#                 filepath='weights-{epoch:02d}-{val_loss:.2f}.hdf5', \
#                 monitor='val_loss', verbose=0, save_best_only=False, \
#                 save_weights_only=False, mode='auto')


# train the model, output generated text after each iteration
for iteration in range(1, 2):
# for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # model.fit(X, y, batch_size=128, nb_epoch=800, callbacks=[checkpoint])
    model.fit(X, y, batch_size=128, nb_epoch=10000, callbacks=[checkpoint])

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2, 2.0, 3.0, 5.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(1500):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
print()
