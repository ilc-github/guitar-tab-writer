'''
Modified version of the keras docs lstm_text_generation example code.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


'''
Using base LSTM text generation example from Keras repo
'''

# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
# path = '../data/mini_input.txt'
# path = '../data/skyrim_repeat.txt'
# text = open(path).read().lower()
# print('corpus length:', len(text))

# chars = sorted(list(set(text)))
# print('total chars:', len(chars))
# char_indices = dict((c, i) for i, c in enumerate(chars))
# indices_char = dict((i, c) for i, c in enumerate(chars))

chars = ['\n', '%', '-', '0', '1', '2', '3', '4', '5', '6', \
                        '7', '8', '9', '|']
char_indices = dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i,c) for i, c in enumerate(chars))

clean1 = np.load('../data/clean1_np_mat.npy')
# clean2 = np.load('clean2_np_mat.npy')

# mat = np.concatenate((clean1, clean2), axis=0).astype(np.bool)
mat = clean1

# sample from numpy matrix in semi-redundant sequences of maxlen time-steps
maxlen = 40
step = 2
timesteps_windows = []
next_timestep = []
for i in xrange(0, len(mat) - maxlen, step):
    timesteps_windows.append(mat[i: i + maxlen])
    next_timestep.append(mat[i + maxlen])
print('nb sequences:', len(timesteps_windows))

print('Vectorization...')
X = np.zeros((len(timesteps_windows), maxlen, len(chars) * 6), dtype=np.bool)
y = np.zeros((len(timesteps_windows), len(chars) * 6), dtype=np.bool)
for i, timesteps in enumerate(timesteps_windows):
    for t, step in enumerate(timesteps):
        for j, val in enumerate(step):
            one_hot_index = None
            if j > 13:
                one_hot_index = j % 14
            else:
                one_hot_index = j
            # X[i, t, char_indices[one_hot_index]] = np.bool(val)
            X[i, t, j] = np.bool(val)

for i, timestep in enumerate(next_timestep):
    for j, val in enumerate(timestep):
        y[i, char_indices[j % 13 - 1]] = val



# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.compile(loss='categorical_crossentropy', optimizer='adam')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(mat) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = mat[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
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
