import numpy as np

''' Modify starter project data (500mb cleaned guitar tabs)
 * create smaller dataset for experimentation (~300k lines)
 * create pre-processed input file (alternate input layout; reduce character
 class imbalances)
'''

path_large = '../data/input.txt'
path_write_mini = '../data/mini_input.txt'

# with open(path_large, 'r') as f:
#     ans = []
#     for line in f:
#         ans.append(line)
#
# with open(path_write_mini, 'w') as f:
#     for item in ans[:300000]:
#       f.write("%s" % item)

# ''' Write 15k lines of tabs
# '''
# with open('mini_1mb_input.txt', 'w') as f:
#     for item in ans[:15000]:
#       f.write("%s" % item)

with open(path_large, 'r') as f:
    ans = []
    for line in f:
        ans.append(line)

ans[0] = ans[0][7:] # remove random symbols from first line
mini = ans[:48]
one = ans[:5]
two = ans[6:12]

check = []
for i in xrange(2, len(two[0])):
    # print two[0][i]
    check.append(two[0][i])

chars = ['\n', '%', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '|']
char_indices = dict((c,i) for i, c in enumerate(chars))
indices_char = dict((i,c) for i, c in enumerate(chars))

''' Vectorize of one block (block 'two')
'''
two
curr_len = len(two[0])
steps = curr_len - 1

'''
[Step 1]
Produce list of vectorized chars by time step for 1st string E
'''
vt = [[] for _ in xrange(steps)]

for i, char in enumerate(two[0][1:]):
    vectorize_char = np.zeros(len(chars), dtype=np.int)
    vectorize_char[char_indices[char]] = 1
    vt[i].extend(list(vectorize_char))

'''
[Step 2]
Extend list of vectorized chars in Step 1 with vec chars for string B
'''
vt_2_test = vt
for i, char in enumerate(two[1][1:]):
    vectorize_char = np.zeros(len(chars), dtype=np.int)
    vectorize_char[char_indices[char]] = 1
    vt_2_test[i].extend(list(vectorize_char))

'''
[All-in-one]
Produce list of vectorized chars by time step for 1st string E,
then extend the list with vectorized chars for the remaining
5 strings.
'''
vt_check = [[] for _ in xrange(steps)]

for string_num in xrange(6):
    for i, char in enumerate(two[string_num][1:]):
        vectorize_char = np.zeros(len(chars), dtype=np.int)
        vectorize_char[char_indices[char]] = 1
        vt_check[i].extend(list(vectorize_char))



print('Vectorization of one line')
X = np.zeros(num_time_steps, curr_len**will change across each block**, len(chars))
# X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((num_time_steps, len(chars)), dtype=np.bool)
for i, step in enumerate(num_time_steps):
# for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

'''
vt = [[] for _ in len(until '\n')]
for i, char in enumerate(txt):
    vt[i].extend(vectorize(char))  # i = time step
'''
