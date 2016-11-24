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
    txt = txt.replace('\n\n\n\n\n\n.', '')
    txt = txt.replace('||||||.', '')
    strings = [txt[i::7] for i in xrange(6)]
    return strings[0], strings[1], strings[2], strings[3], strings[4], strings[5]

def prep_flat_nb_int(filepath):
    '''
    Prep for model integer input formatting
    INPUT:  String (filepath)
    OUTPUT: 2d numpy array (timestep x string 1 - 6 codified tab values)
    '''
    with open(filepath) as f:
        txt = f.read()
    if len(txt) % 7 != 0:
        print('Check flat input text file formatting')
        return
    txt = txt.replace('\n\n\n\n\n\n.', '')
    txt = txt.replace('||||||.', '')
    for i in range(10)[::-1]:
        if i == 9:
            txt = txt.replace(str(i), '*')
        else:
            txt = txt.replace(str(i), str(i+1))
    txt = txt.replace('-', '0')
    strings = [txt[i::7] for i in xrange(6)]
    strings_mat = np.zeros((len(strings[0]), 6))
    for i in xrange(len(strings[0])):
        for j in xrange(len(strings)):
            if strings[j][i] == '*':
                strings_mat[i][j] = 10
            else:
                strings_mat[i][j] = strings[j][i]
    return strings_mat




# s1, s2, s3, s4, s5, s6 = prep_flat_nb(text_filepath)
# text = s1 + s2 + s3 + s4 + s5 + s6

strings_mat = prep_flat_nb_int(text_filepath)


print('total # timesteps:'), strings_mat.shape[0]


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

def vectorize_int(mat, maxlen=40, step=3):
    '''
    INPUT: 2d numpy array (timestep x string 1 - 6 codified tab values)
    OUTPUT: 3d numpy array (vectorized X), 2d numpy array (vectorized y)
    '''
    # cut the text in semi-redundant sequences of maxlen characters
    timestep_range = []
    next_step = []
    for i in range(0, mat.shape[0] - maxlen, step):
        timestep_range.append(mat[i: i + maxlen])
        next_step.append(mat[i + maxlen])

    # Vectorization
    X = np.zeros((len(timestep_range), maxlen, 6))
    y = np.zeros((len(timestep_range), 6))
    for i, timesteps in enumerate(timestep_range):
        for t in xrange(timesteps.shape[0]):
            for j, tstep in enumerate(timesteps[t]):
                X[i, t, j] = tstep
        for k, tstep in enumerate(next_step[i]):
            y[i, k] = tstep
        # y[i, char_indices[next_step[i]]] = 1
    return X, y

X, y = vectorize_int(strings_mat)

''' Set maxlen
'''
maxlen = 40

# Build model using functional API
# inputs: receive sequences of 40 integers,
print('Build model via functional API...')
str_all = Input(shape=(maxlen, 6), name='input_1')
# str_2 = Input(shape=(maxlen, num_chars), name='input_2')
# str_3 = Input(shape=(maxlen, num_chars), name='input_3')
# str_4 = Input(shape=(maxlen, num_chars), name='input_4')
# str_5 = Input(shape=(maxlen, num_chars), name='input_5')
# str_6 = Input(shape=(maxlen, num_chars), name='input_6')

shared_lstm_1 = LSTM(128, return_sequences=True)

encoded_str_all_ = shared_lstm_1(str_all)
# encoded_str2_ = shared_lstm_1(str_2)
# encoded_str3_ = shared_lstm_1(str_3)
# encoded_str4_ = shared_lstm_1(str_4)
# encoded_str5_ = shared_lstm_1(str_5)
# encoded_str6_ = shared_lstm_1(str_6)

shared_lstm_2 = LSTM(128)

encoded_str_all = shared_lstm_2(encoded_str_all_)
# encoded_str2 = shared_lstm_2(encoded_str2_)
# encoded_str3 = shared_lstm_2(encoded_str3_)
# encoded_str4 = shared_lstm_2(encoded_str4_)
# encoded_str5 = shared_lstm_2(encoded_str5_)
# encoded_str6 = shared_lstm_2(encoded_str6_)

output_str_all = Dense(len(chars), activation='relu', name='output_str_all')(encoded_str_all)
# output_str2 = Dense(len(chars), activation='softmax', name='output_str2')(encoded_str2)
# output_str3 = Dense(len(chars), activation='softmax', name='output_str3')(encoded_str3)
# output_str4 = Dense(len(chars), activation='softmax', name='output_str4')(encoded_str4)
# output_str5 = Dense(len(chars), activation='softmax', name='output_str5')(encoded_str5)
# output_str6 = Dense(len(chars), activation='softmax', name='output_str6')(encoded_str6)

# model = Model(input=[str_1, str_2, str_3, str_4, str_5, str_6], \
#              output=[output_str1, output_str2, output_str3, output_str4, \
#                      output_str5, output_str6])

model = Model(input=[str_all], output=[output_str_all])

optimizer = RMSprop(lr=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# fit_X = [X1, X2, X3, X4, X5, X6]
# fit_label = [y1, y2, y3, y4, y5, y6]

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
    model.fit(X, y, batch_size=128, nb_epoch=800, callbacks=[checkpoint])


    # model.fit(fit_X, fit_label, batch_size=1024, nb_epoch=800, \
    #                                         callbacks=[checkpoint])
    # model.fit(fit_X, fit_label, batch_size=1024, nb_epoch=800)
    # model.save('model_fn_api_skyrim.h5')

    start_index = random.randint(0, strings_mat.shape[0] - maxlen - 1)
    output_text = ''
    for diversity in [0.2, 0.5, 1.0, 1.2, 2.0, 3.0, 5.0, 100.0, 1000.0, 100000.0]:
        output_text += '\n'
        output_text += '----- diversity:' + str(diversity) + '\n'

        generated1 = ''
        # generated2 = ''
        # generated3 = ''
        # generated4 = ''
        # generated5 = ''
        # generated6 = ''
        all_timestep = strings_mat[start_index: start_index + maxlen]


        # s2_timestep = s2[start_index: start_index + maxlen]
        # s3_timestep = s3[start_index: start_index + maxlen]
        # s4_timestep = s4[start_index: start_index + maxlen]
        # s5_timestep = s5[start_index: start_index + maxlen]
        # s6_timestep = s6[start_index: start_index + maxlen]
        generated1 += all_timestep
        # generated2 += s2_timestep
        # generated3 += s3_timestep
        # generated4 += s4_timestep
        # generated5 += s5_timestep
        # generated6 += s6_timestep
        # print('----- Generating with seed (string 1): "' + s1_timestep + '"')
        output_text += '----- Generating with seed (string 1): "' + \
                                                    str(all_timestep) + '"' + '\n'
        # sys.stdout.write(generated1)
        # print('----- Generating with seed (string 2): "' + s2_timestep + '"')
        # output_text += '----- Generating with seed (string 1): "' + \
        #                                             s2_timestep + '"' + '\n'
        # sys.stdout.write(generated2)
        # print('----- Generating with seed (string 3): "' + s3_timestep + '"')
        # output_text += '----- Generating with seed (string 1): "' + \
        #                                             s3_timestep + '"' + '\n'
        # sys.stdout.write(generated3)
        # print('----- Generating with seed (string 4): "' + s4_timestep + '"')
        # output_text += '----- Generating with seed (string 1): "' + \
        #                                             s4_timestep + '"' + '\n'
        # sys.stdout.write(generated4)
        # print('----- Generating with seed (string 5): "' + s5_timestep + '"')
        # output_text += '----- Generating with seed (string 1): "' + \
        #                                             s5_timestep + '"' + '\n'
        # sys.stdout.write(generated5)
        # print('----- Generating with seed (string 6): "' + s6_timestep + '"')
        # output_text += '----- Generating with seed (string 1): "' + \
        #                                             s6_timestep + '"' + '\n'
        # sys.stdout.write(generated6)

        '''  Stopped here-- need a way to handle the new prediction format
        '''

        # tab_chunk = ''
        for i in range(200):
            x = np.zeros((1, maxlen, len(chars)))
            # x2 = np.zeros((1, maxlen, len(chars)))
            # x3 = np.zeros((1, maxlen, len(chars)))
            # x4 = np.zeros((1, maxlen, len(chars)))
            # x5 = np.zeros((1, maxlen, len(chars)))
            # x6 = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(s1_timestep):
                x[0, t, char_indices[char]] = 1.
            # for t, char in enumerate(s2_timestep):
            #     x2[0, t, char_indices[char]] = 1.
            # for t, char in enumerate(s3_timestep):
            #     x3[0, t, char_indices[char]] = 1.
            # for t, char in enumerate(s4_timestep):
            #     x4[0, t, char_indices[char]] = 1.
            # for t, char in enumerate(s5_timestep):
            #     x5[0, t, char_indices[char]] = 1.
            # for t, char in enumerate(s6_timestep):
            #     x6[0, t, char_indices[char]] = 1.

            # pred_x = [x1, x2, x3, x4, x5, x6]
            # preds = model.predict(pred_x, verbose=0)[0]
            # preds = model.predict(pred_x, verbose=0)
            preds = model.predict(x, verbose=0)[0]
            next_index1 = sample(preds, diversity)
            # next_index2 = sample(preds[1][0], diversity)
            # next_index3 = sample(preds[2][0], diversity)
            # next_index4 = sample(preds[3][0], diversity)
            # next_index5 = sample(preds[4][0], diversity)
            # next_index6 = sample(preds[5][0], diversity)
            next_char1 = indices_char[next_index1]
            # next_char2 = indices_char[next_index2]
            # next_char3 = indices_char[next_index3]
            # next_char4 = indices_char[next_index4]
            # next_char5 = indices_char[next_index5]
            # next_char6 = indices_char[next_index6]

            generated1 += next_char1
            # generated2 += next_char2
            # generated3 += next_char3
            # generated4 += next_char4
            # generated5 += next_char5
            # generated6 += next_char6
            s1_timestep = s1_timestep[1:] + next_char1
            # s2_timestep = s2_timestep[1:] + next_char2
            # s3_timestep = s3_timestep[1:] + next_char3
            # s4_timestep = s4_timestep[1:] + next_char4
            # s5_timestep = s5_timestep[1:] + next_char5
            # s6_timestep = s6_timestep[1:] + next_char6
            output_text += next_char1 + next_char2 + next_char3 + next_char4 + \
                                        next_char5 + next_char6 + '.'

    if not os.path.exists('output/'):
        os.makedirs('output/')
    with open('output/output_tab' + str(iteration) + '.txt', 'w') as f:
        f.write(output_text)
