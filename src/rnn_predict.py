'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os


'''
Using base LSTM text generation example from Keras repo
'''

if len(sys.argv) < 3:
    sys.exit('Usage: %s weights-filepath.hdf5 flat_tab_input.txt' % sys.argv[0])

if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: Weights-filepath.hdf5 %s was not found!' % sys.argv[1])
elif not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: flat_tab_input.txt %s was not found!' % sys.argv[2])

weights_filepath = sys.argv[1]
flat_tab_filepath = sys.argv[2]

path = flat_tab_filepath
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40

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


model.load_weights(weights_filepath)

for iteration in range(1, 2):
    output_text = ''
    output_text += '\n'
    output_text += '-' * 50 + '\n'
    output_text += 'Iteration' + str(iteration) + '\n'

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2, 2.0, 3.0]: # testing 2.0 diversity
        output_text += '\n'
        output_text += '----- diversity:' + str(diversity) + '\n'

        generated = ''
        # start_index = 0 # checking if reproduces intro
        seed_tab = text[start_index: start_index + maxlen]
        generated += seed_tab
        output_text += '----- Generating with seed: "' + seed_tab + '"' + '\n'
        # output_text += generated

        tab_chunk = ''
        for i in range(1500):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(seed_tab):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            # generated += next_char
            seed_tab = seed_tab[1:] + next_char
            tab_chunk += next_char
        output_text += tab_chunk

    if not os.path.exists('output/'):
        os.makedirs('output/')
    with open('output/output_tab' + str(iteration) + '.txt', 'w') as f:
        f.write(output_text)



test = '||||||.------.-0--2-.------.---4--.-----||||||.------.-0--2-.------.---4--.------.--44--.------.-0----.------.--2---.------.------.------.--2---.------.2-----.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.------.-----7.------.------.------.0-----.------.||||||.------.-2--0-.------.-3----.------.--2---.------.-3----.------.--2---.------.2-----.------.------.------.------.------.------.------.------'
