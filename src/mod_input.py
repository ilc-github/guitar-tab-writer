import numpy as np

''' Modify starter project data (500mb cleaned guitar tabs)
 * create smaller dataset for experimentation (~300k lines)
 * create pre-processed input file (alternate input layout; reduce character
 class imbalances)
'''

# with open(path_write_mini, 'w') as f:
#     for item in ans[:300000]:
#       f.write("%s" % item)

# with open('mini_1mb_input.txt', 'w') as f:
#     for item in ans[:15000]:
#       f.write("%s" % item)


class ProcessInputs(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.chars = ['\n', '%', '-', '0', '1', '2', '3', '4', '5', '6', \
                                '7', '8', '9', '|']
        self.char_indices = dict((c,i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i,c) for i, c in enumerate(self.chars))

    def gen_list(self):
        ans = []
        with open(self.filepath, 'r') as f:
            for line in f:
                ans.append(line)
        return ans

    def vectorize_one_chunk(self, list_text):
        '''
        INPUT:  list containing six lines of text (ordered as E, B, G, D, A, E
                strings)
        OUTPUT: list of lists, each row representing a time-step (column of single
                tab values, one for each of six strings E-B-G-D-A-E); columns
                represent one-hot according to char_indices
        '''
        steps = len(list_text[0]) - 1
        vt = [[] for _ in xrange(steps)]

        for string_num in xrange(6):
            for i, char in enumerate(list_text[string_num][1:]):
                vectorize_char = np.zeros(len(self.chars), dtype=np.int)
                vectorize_char[self.char_indices[char]] = 1
                vt[i].extend(list(vectorize_char))
        return vt

    def vectorize_all(self, list_text):
        vt_all = []
        while True:
            if (list_text[0][0] == 'E') and (list_text[1][0] == 'B'):
                tab_chunk = list_text[0:6]
                vt_temp = self.vectorize_one_chunk(tab_chunk)
                vt_all += vt_temp
                if len(list_text) < 7:
                    break
                else:
                    list_text = list_text[7:]
                    if len(list_text) == 0:
                        break
            else:
                break
        return vt_all


    # if group[0][0] == 'E' and group[0+1][0] == 'B':
    #     ''' create tab chunk'''
    #     chunk = group[0:6]
    #     new_vt = inputs.vectorize_one_chunk(chunk)
    #     ''' set group = group sliced down to the section that was not checked '''
    #     test = group[7:]
    #     ''' if no more upcoming line 'E' + line 'B' after it, stop '''
    #     '''repeat until no more'''

'''
[Next step]
Check if can create list of np arrays instead (as opposed to list of lists)
Add in layer for handling %, and moving to a new "chunk" (will require
setting the new steps length for vt list declaration)
'''


''' Vectorize multiple blocks
'''

# if (group[0][0] == 'E') and (group[1][0] == 'B'):
    # pass





print('Vectorization of one line')
''' 'curr_len' will change across each block
'''
# X = np.zeros(num_time_steps, curr_len, len(chars))
# y = np.zeros((num_time_steps, len(chars)), dtype=np.bool)
# for i, step in enumerate(num_time_steps):
#     for t, char in enumerate(step):
#         X[i, t, char_indices[char]] = 1
#     y[i, char_indices[next_chars[i]]] = 1

'''
vt = [[] for _ in len(until '\n')]
for i, char in enumerate(txt):
    vt[i].extend(vectorize(char))  # i = time step
'''


if __name__ == '__main__':
    inputs = ProcessInputs('../data/input.txt')
    ans = inputs.gen_list()

    ans[0] = ans[0][7:] # remove random symbols from first line
    two = ans[6:12]

    vt = inputs.vectorize_one_chunk(two)
    print "Len of vt = ", len(vt)
    print "Len of vt[0] = ", len(vt[0])

    group = ans[6:27]
    mini = ans[:48]

    # vt_group = inputs.vectorize_all(ans[6:1000005])

    ''' Check if 6 rows between each '%' marker
    '''
    # from collections import Counter
    # temp = Counter()
    # for i in xrange(len(ans)):
    #     temp.update(ans[i][0])
    # print temp

    ''' Count # of 6-line chunks that do not have uniform line length
    '''

    cnt = 0
    num_chunks = 0

    chunk_len = 0
    uneven_flag = None
    num_uneven_chunks = 0
    for i in xrange(len(ans)):
        if uneven_flag:
            ''' First pass
            '''
            if ans[i][0] != '%':
                cnt += 1

            else:
                if cnt
        else:
            ''' Second pass
            '''

    # will likely need to drop all chunks where line length not uniform
