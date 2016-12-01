'''
Modified version of the keras docs lstm_text_generation example code.
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


if len(sys.argv) < 2:
    sys.exit('Usage: %s flat_tab_input.txt' % sys.argv[0])

if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: flat_tab_input.txt %s was not found!' % sys.argv[1])

text_filepath = sys.argv[1]



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
s1, s2, s3, s4, s5, s6 = prep_flat_nb(text_filepath)
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

def vectorize(string_txt, maxlen=40, step=3):
    '''
    INPUT: string containing tab characters for 1 guitar string
    OUTPUT: 3d numpy array (vectorized X), 2d numpy array (vectorized y)
    '''
    # cut the text in semi-redundant sequences of maxlen characters
    timesteps = []
    next_chars = []
    for i in range(0, len(string_txt) - maxlen, step):
        timesteps.append(string_txt[i: i + maxlen])
        next_chars.append(string_txt[i + maxlen])

    # Vectorization
    X = np.zeros((len(timesteps), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(timesteps), len(chars)), dtype=np.bool)
    for i, timestep in enumerate(timesteps):
        for t, char in enumerate(timestep):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y

''' Set maxlen
'''
maxlen = 60

X1, y1 = vectorize(s1, maxlen, 1)
X2, y2 = vectorize(s2, maxlen, 1)
X3, y3 = vectorize(s3, maxlen, 1)
X4, y4 = vectorize(s4, maxlen, 1)
X5, y5 = vectorize(s5, maxlen, 1)
X6, y6 = vectorize(s6, maxlen, 1)



# Build model using functional API
# inputs: receive sequences of 40 integers,
print('Build model via functional API...')
str_1 = Input(shape=(maxlen, num_chars), name='input_1') # name these, will show up in summary
str_2 = Input(shape=(maxlen, num_chars), name='input_2')
str_3 = Input(shape=(maxlen, num_chars), name='input_3')
str_4 = Input(shape=(maxlen, num_chars), name='input_4')
str_5 = Input(shape=(maxlen, num_chars), name='input_5')
str_6 = Input(shape=(maxlen, num_chars), name='input_6')

shared_lstm_1 = LSTM(128, return_sequences=True)

encoded_str1_ = shared_lstm_1(str_1)
encoded_str2_ = shared_lstm_1(str_2)
encoded_str3_ = shared_lstm_1(str_3)
encoded_str4_ = shared_lstm_1(str_4)
encoded_str5_ = shared_lstm_1(str_5)
encoded_str6_ = shared_lstm_1(str_6)

shared_lstm_2 = LSTM(128)

encoded_str1 = shared_lstm_2(encoded_str1_)
encoded_str2 = shared_lstm_2(encoded_str2_)
encoded_str3 = shared_lstm_2(encoded_str3_)
encoded_str4 = shared_lstm_2(encoded_str4_)
encoded_str5 = shared_lstm_2(encoded_str5_)
encoded_str6 = shared_lstm_2(encoded_str6_)

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

fit_X = [X1, X2, X3, X4, X5, X6]
fit_label = [y1, y2, y3, y4, y5, y6]

# ''' build the model: a single LSTM
# '''
# print('Build model...')
# model = Sequential()
# model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
# model.add(LSTM(128, input_shape=(maxlen, len(chars))))
# model.add(Dense(len(chars)))
# model.add(Activation('softmax'))
#
# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) # CHECK: when temp low, all of these should be approx same.  when high, all but one should be approx 0 (one is close to 1)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

checkpoint = ModelCheckpoint( \
                filepath='weights-{epoch:02d}-{loss:.2f}.hdf5', \
                monitor='loss', verbose=0, save_best_only=True, \
                save_weights_only=False, mode='auto')


# train the model, output generated text after each iteration
for iteration in range(1, 2):
# for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # model.fit(X, y, batch_size=128, nb_epoch=800, callbacks=[checkpoint])
    model.fit(fit_X, fit_label, batch_size=2048, nb_epoch=800, \
                                            callbacks=[checkpoint])
    # model.fit(fit_X, fit_label, batch_size=1024, nb_epoch=800)
    # model.save('model_fn_api_skyrim.h5')

    start_index = random.randint(0, len(s1) - maxlen - 1)
    output_text = ''
    for diversity in [0.2, 0.5, 1.0, 1.2, 2.0, 3.0, 5.0, 10.0]:
        output_text += '\n'
        output_text += '----- diversity:' + str(diversity) + '\n'

        generated1 = ''
        generated2 = ''
        generated3 = ''
        generated4 = ''
        generated5 = ''
        generated6 = ''
        s1_timestep = s1[start_index: start_index + maxlen]
        s2_timestep = s2[start_index: start_index + maxlen]
        s3_timestep = s3[start_index: start_index + maxlen]
        s4_timestep = s4[start_index: start_index + maxlen]
        s5_timestep = s5[start_index: start_index + maxlen]
        s6_timestep = s6[start_index: start_index + maxlen]
        generated1 += s1_timestep
        generated2 += s2_timestep
        generated3 += s3_timestep
        generated4 += s4_timestep
        generated5 += s5_timestep
        generated6 += s6_timestep
        # print('----- Generating with seed (string 1): "' + s1_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    s1_timestep + '"' + '\n'
        # sys.stdout.write(generated1)
        # print('----- Generating with seed (string 2): "' + s2_timestep + '"')
        output_text += '----- Generating with seed (string 2): "' + \
                                                    s2_timestep + '"' + '\n'
        # sys.stdout.write(generated2)
        # print('----- Generating with seed (string 3): "' + s3_timestep + '"')
        output_text += '----- Generating with seed (string 3): "' + \
                                                    s3_timestep + '"' + '\n'
        # sys.stdout.write(generated3)
        # print('----- Generating with seed (string 4): "' + s4_timestep + '"')
        output_text += '----- Generating with seed (string 4): "' + \
                                                    s4_timestep + '"' + '\n'
        # sys.stdout.write(generated4)
        # print('----- Generating with seed (string 5): "' + s5_timestep + '"')
        output_text += '----- Generating with seed (string 5): "' + \
                                                    s5_timestep + '"' + '\n'
        # sys.stdout.write(generated5)
        # print('----- Generating with seed (string 6): "' + s6_timestep + '"')
        output_text += '----- Generating with seed (string 6): "' + \
                                                    s6_timestep + '"' + '\n'
        # sys.stdout.write(generated6)

        # tab_chunk = ''
        for i in range(200):
            x1 = np.zeros((1, maxlen, len(chars)))
            x2 = np.zeros((1, maxlen, len(chars)))
            x3 = np.zeros((1, maxlen, len(chars)))
            x4 = np.zeros((1, maxlen, len(chars)))
            x5 = np.zeros((1, maxlen, len(chars)))
            x6 = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(s1_timestep):
                x1[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s2_timestep):
                x2[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s3_timestep):
                x3[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s4_timestep):
                x4[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s5_timestep):
                x5[0, t, char_indices[char]] = 1.
            for t, char in enumerate(s6_timestep):
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
            output_text += next_char1 + next_char2 + next_char3 + next_char4 + \
                                        next_char5 + next_char6 + '.'

    if not os.path.exists('output/'):
        os.makedirs('output/')
    with open('output/output_tab' + str(iteration) + '.txt', 'w') as f:
        f.write(output_text)
