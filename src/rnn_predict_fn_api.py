'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import LSTM, Embedding, merge
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
'''
******NOTE****** need to re-do (lstm_fn_api.py base has changed)
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


'''
Using base LSTM text generation example from Keras repo
'''

def prep_flat_nb(filepath):
    with open(filepath) as f:
        txt = f.read()
    if len(txt) % 7 != 0:
        print('Check flat input text file formatting')
        return
    string1 = txt[::7]
    string2 = txt[1::7]
    string3 = txt[2::7]
    string4 = txt[3::7]
    string5 = txt[4::7]
    string6 = txt[5::7]
    return string1, string2, string3, string4, string5, string6

# s1, s2, s3, s4, s5, s6 = prep_flat_nb('../data/flat_skyrim_nb.txt')
s1, s2, s3, s4, s5, s6 = prep_flat_nb(path)
# path = text_filepath
# text = open(path).read().lower()

text = s1 + s2 + s3 + s4 + s5 + s6
print('total character length (all strings):'), len(text)
print('number timesteps:', len(s1))

chars = sorted(list(set(text)))
num_chars = len(chars)
print('total chars:', num_chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


''' Set maxlen
'''
maxlen = 40

# Build model using functional API
# inputs: receive sequences of 40 integers,
print('Build model via functional API...')
str_1 = Input(shape=(maxlen, num_chars))
str_2 = Input(shape=(maxlen, num_chars))
str_3 = Input(shape=(maxlen, num_chars))
str_4 = Input(shape=(maxlen, num_chars))
str_5 = Input(shape=(maxlen, num_chars))
str_6 = Input(shape=(maxlen, num_chars))

shared_lstm = LSTM(128)

encoded_str1 = shared_lstm(str_1)
encoded_str2 = shared_lstm(str_2)
encoded_str3 = shared_lstm(str_3)
encoded_str4 = shared_lstm(str_4)
encoded_str5 = shared_lstm(str_5)
encoded_str6 = shared_lstm(str_6)

output_str1 = Dense(len(chars), activation='softmax', name='output_str1')(encoded_str1)
output_str2 = Dense(len(chars), activation='softmax', name='output_str2')(encoded_str2)
output_str3 = Dense(len(chars), activation='softmax', name='output_str3')(encoded_str3)
output_str4 = Dense(len(chars), activation='softmax', name='output_str4')(encoded_str4)
output_str5 = Dense(len(chars), activation='softmax', name='output_str5')(encoded_str5)
output_str6 = Dense(len(chars), activation='softmax', name='output_str6')(encoded_str6)

model = Model(input=[str_1, str_2, str_3, str_4, str_5, str_6], \
             output=[output_str1, output_str2, output_str3, output_str4, \
                     output_str5, output_str6])

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', \
              loss_weights=[1., 1., 1., 1., 1., 1.])

model.load_weights(weights_filepath)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


for iteration in range(1, 2):
# for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # model.fit(fit_X, fit_label, batch_size=128, nb_epoch=800, \
    #                                         callbacks=[checkpoint])

    start_index = random.randint(0, len(text) - maxlen - 1)
    output_text = ''
    for diversity in [0.2, 0.5, 1.0, 1.2, 2.0, 3.0, 5.0]:
        output_text = '\n'
        output_text += '----- diversity:' + str(diversity) + '\n'

        generated1 = ''
        generated2 = ''
        generated3 = ''
        generated4 = ''
        generated5 = ''
        generated6 = ''
        s1_timestep = text[start_index: start_index + maxlen]
        s2_timestep = text[start_index: start_index + maxlen]
        s3_timestep = text[start_index: start_index + maxlen]
        s4_timestep = text[start_index: start_index + maxlen]
        s5_timestep = text[start_index: start_index + maxlen]
        s6_timestep = text[start_index: start_index + maxlen]
        generated1 += s1_timestep
        generated2 += s1_timestep
        generated3 += s1_timestep
        generated4 += s1_timestep
        generated5 += s1_timestep
        generated6 += s1_timestep
        # print('----- Generating with seed (string 1): "' + s1_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    s1_timestep + '"' + '\n'
        # sys.stdout.write(generated1)
        # print('----- Generating with seed (string 2): "' + s2_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    s2_timestep + '"' + '\n'
        # sys.stdout.write(generated2)
        # print('----- Generating with seed (string 3): "' + s3_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    s3_timestep + '"' + '\n'
        # sys.stdout.write(generated3)
        # print('----- Generating with seed (string 4): "' + s4_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    s4_timestep + '"' + '\n'
        # sys.stdout.write(generated4)
        # print('----- Generating with seed (string 5): "' + s5_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    s5_timestep + '"' + '\n'
        # sys.stdout.write(generated5)
        # print('----- Generating with seed (string 6): "' + s6_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    s6_timestep + '"' + '\n'
        # sys.stdout.write(generated6)

        tab_chunk = ''
        for i in range(50):
            x1 = np.zeros((1, maxlen, len(chars)))
            x2 = np.zeros((1, maxlen, len(chars)))
            x3 = np.zeros((1, maxlen, len(chars)))
            x4 = np.zeros((1, maxlen, len(chars)))
            x5 = np.zeros((1, maxlen, len(chars)))
            x6 = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(s1_timestep):
                x1[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s1_timestep):
                x2[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s1_timestep):
                x3[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s1_timestep):
                x4[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s1_timestep):
                x5[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s1_timestep):
                x6[0, t, char_indices[char]] = 1.

            pred_x = [x1, x2, x3, x4, x5, x6]
            # preds = model.predict(pred_x, verbose=0)[0]
            preds = model.predict(pred_x, verbose=0)
            next_index1 = sample(preds[0][0], diversity)
            next_index2 = sample(preds[1][0], diversity)
            next_index3 = sample(preds[2][0], diversity)
            next_index4 = sample(preds[3][0], diversity)
            next_index5 = sample(preds[4][0], diversity)
            next_index6 = sample(preds[5][0], diversity)
            next_char1 = indices_char[next_index1]
            next_char2 = indices_char[next_index2]
            next_char3 = indices_char[next_index3]
            next_char4 = indices_char[next_index4]
            next_char5 = indices_char[next_index5]
            next_char6 = indices_char[next_index6]

            generated1 += next_char1
            generated2 += next_char2
            generated3 += next_char3
            generated4 += next_char4
            generated5 += next_char5
            generated6 += next_char6
            s1_timestep = s1_timestep[1:] + next_char1
            s2_timestep = s2_timestep[1:] + next_char2
            s3_timestep = s3_timestep[1:] + next_char3
            s4_timestep = s4_timestep[1:] + next_char4
            s5_timestep = s5_timestep[1:] + next_char5
            s6_timestep = s6_timestep[1:] + next_char6
            tab_chunk += next_char1 + next_char2 + next_char3 + next_char4 + \
                                        next_char5 + next_char6 + '.'
        output_text += tab_chunk

            # sys.stdout.write(next_char1)
            # sys.stdout.write(next_char2)
            # sys.stdout.write(next_char3)
            # sys.stdout.write(next_char4)
            # sys.stdout.write(next_char5)
            # sys.stdout.write(next_char6)
            # sys.stdout.write('.')
            # sys.stdout.flush()
    if not os.path.exists('output/'):
        os.makedirs('output/')
    with open('output/output_tab' + str(iteration) + '.txt', 'w') as f:
        f.write(output_text)
