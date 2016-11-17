import numpy as np
import pandas as pd
from collections import Counter
import time


''' Modify starter project data (500mb guitar tabs)
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
        ''' Preprocessing input text file
        INPUT: input text file path
        '''
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

    '''
    vt = [[] for _ in len(until '\n')]
    for i, char in enumerate(txt):
        vt[i].extend(vectorize(char))  # i = time step
    '''

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

    # def vectorize_all_old(self, list_text):
    #     vt_all = []
    #     while True:
    #         if (list_text[0][0] == 'E') and (list_text[1][0] == 'B'):
    #             tab_chunk = list_text[0:6]
    #             vt_temp = self.vectorize_one_chunk(tab_chunk)
    #             vt_all += vt_temp
    #             if len(list_text) < 7:
    #                 break
    #             else:
    #                 list_text = list_text[7:]
    #                 if len(list_text) == 0:
    #                     break
    #         else:
    #             break
    #     return vt_all

    def vectorize_all(self, list_text):
        vt_all = []
        i = 0
        while len(list_text) >= 1 + i * 7:
            if (list_text[0 + i * 7][0] == 'E') and \
                                (list_text[1 + i * 7][0] == 'B'):
                lower, upper = (0 + i * 7, 6 + i * 7)
                vt_temp = self.vectorize_one_chunk(list_text[lower:upper])
                vt_all += vt_temp
                if len(list_text[7 * i:]) < 7:
                    break
                else:
                    # list_text = list_text[7:]
                    i += 1
                    if len(list_text[7 * i:]) == 0:
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


def check_line_len_uniform(list_text): # SLOW, procedural list format
    '''
    Check if line length is uniform--prints out # of problematic tab chunks
    (sets of 6 lines, separated by '%')

    INPUT: list of text, lines of guitar tabs
    OUTPUT: None
    '''
    num_chunks, num_uneven, num_even = (0, 0, 0)
    chunk_len, uneven_flag = (-1, 0)
    for i in xrange(len(list_text)):
        if chunk_len > 0:
            if list_text[i][0] != '%': # Second pass
                if len(list_text[i]) != chunk_len:
                    uneven_flag = 1
            else: # evaluate chunk
                if uneven_flag == 1:
                    num_uneven += 1
                    uneven_flag = 0
                    chunk_len = -1
                else:
                    num_even += 1
                    chunk_len = -1
        else:
            if chunk_len == -1: # First pass
                chunk_len = len(list_text[i])
    print "Number of tab chunks with uneven line lengths = ", num_uneven
    print "Number of tab chunks with even line lengths = ", num_even
    print "% uneven = ", float(num_uneven) / (num_uneven + num_even)

def remove_uneven_chunks_old(list_text): # SLOW, procedural list format
    '''
    Check if line length is uniform for each set of 6 tab lines (each set
    separated by '%').  Removes tab chunks that have uneven line lengths.

    INPUT: list of text, lines of guitar tabs
    OUTPUT: list of text, lines of guitar tabs (cleaned)
    '''
    clean = []
    i = 0
    num_clean, num_del = (0, 0)
    while True:
        if (list_text[0][0] == 'E') and (list_text[1][0] == 'B'): # if on first E string...
            tab_chunk = list_text[0:6]
            flag, len_tracker = (0, 0)

            for line in tab_chunk: # flag = 1 if lines uneven
                if len_tracker == 0:
                    len_tracker = len(line)
                else:
                    if len(line) != len_tracker:
                        flag = 1
            if flag == 0:
                clean += list_text[0:7] # chunk ok, keep
                num_clean += 1
            else:
                num_del += 1 # chunk uneven, exclude from list
            if len(list_text) < 7: # stop while loop if no more tab chunks
                break
            else:
                list_text = list_text[7:]
                print 'Number cleaned chunks = {}, num del = {}'.format(num_clean,num_del)
                if len(list_text) == 0:
                    break
        else:
            break
    return clean, num_clean, num_del # list_text, stripped of uneven lines

def remove_uneven_chunks(list_text): # SLOW, procedural list approach
    '''
    Check if line length is uniform for each set of 6 tab lines (each set
    separated by '%').  Removes tab chunks that have uneven line lengths.

    INPUT: list of text, lines of guitar tabs
    OUTPUT: list of text, lines of guitar tabs (cleaned)
    '''
    clean = []
    i = 0
    num_clean, num_del = (0, 0)
    while len(list_text) >= 1 + i * 7:
        if (list_text[0 + i * 7][0] == 'E') and \
                            (list_text[1 + i * 7][0] == 'B'): # if on first E string...
            flag, len_tracker = (0, 0)
            lower, upper = (0 + i * 7, 6 + i * 7)
            for line in list_text[lower:upper]: # flag = 1 if lines uneven
                if len_tracker == 0:
                    len_tracker = len(line)
                else:
                    if len(line) != len_tracker:
                        flag = 1
            if flag == 0:
                clean += list_text[lower:upper + 1] # chunk ok, keep
                num_clean += 1
            else:
                num_del += 1 # chunk uneven, exclude from list
            if len(list_text[7 * i:]) < 7: # stop while loop if no more tab chunks
                break
            else:
                # list_text = list_text[7:]
                i += 1
                # print 'Number cleaned chunks = {}, num del = {}'.format(num_clean,num_del)
                if len(list_text[7 * i:]) == 0:
                    break
        else:
            break
    return clean, num_clean, num_del # list_text, stripped of uneven lines


def remove_uneven_chunks_numpy(list_text): # attempt later
    '''
    Check if line length is uniform for each set of 6 tab lines (each set
    separated by '%').  Removes tab chunks that have uneven line lengths.

    INPUT: list of text, lines of guitar tabs
    OUTPUT: list of text, lines of guitar tabs (cleaned)
    '''
    clean = []
    i = 0
    num_clean, num_del = (0, 0)
    length = len(list_text)
    while length >= 1 + i * 7:
        # if (list_text[0 + i * 7][0] == 'E') and \
                            # (list_text[1 + i * 7][0] == 'B'): # if on first E string...
        flag, len_tracker = (0, 0)
        lower, upper = (0 + i * 7, 6 + i * 7)
        for line in list_text[lower:upper]: # flag = 1 if lines uneven
            if len_tracker == 0:
                len_tracker = len(line)
            else:
                if len(line) != len_tracker:
                    flag = 1
        if flag == 0:
            clean += list_text[lower:upper + 1] # chunk ok, keep
            num_clean += 1
        else:
            num_del += 1 # chunk uneven, exclude from list
        if length - 7 * i < 7: # stop while loop if no more tab chunks
            break
        else:
            # list_text = list_text[7:]
            i += 1
            # print 'Number cleaned chunks = {}, num del = {}'.format(num_clean,num_del)
            if length - 7 * i == 0:
                break
        # else:
            # break
    return clean, num_clean, num_del # list_text, stripped of uneven lines



def remove_blanks(list_text): # remove time steps where no notes are playing (reduce number of '-' throughout tab inputs)
    '''
    Remove time steps where no notes are playing.  Reduce number of '-'
    throughout tab inputs--produce less sparse input.
    INPUT: list of text, lines of guitar tabs
    OUTPUT: list of text, lines of guitar tabs
    '''
    pass

def check_six_rows(list_text):
    '''
    Check if 6 rows between each '%' marker

    INPUT: List of text, lines of guitar tabs
    OUTPUT: None
    '''
    cnt = Counter()
    for i in xrange(len(list_text)):
        cnt.update(list_text[i][0])
    print cnt

def print_v2n(list_vector, num_notes = 1):
    '''
    Print vector for single note

    INPUT: List of vectorized tab, single time-step
    OUTPUT: None
    '''
    for i in xrange(num_notes):
        print "Time step = ", i + 1
        for j in xrange(0, 84, 14):
            print list_vector[i][j:j + 14]


if __name__ == '__main__':
    inputs = ProcessInputs('../data/input.txt')
    tabs = inputs.gen_list()

    tabs[0] = tabs[0][7:] # remove random symbols from first line
    two = tabs[6:12]

    vt = inputs.vectorize_one_chunk(two)
    print "Len of vt = ", len(vt)
    print "Len of vt[0] = ", len(vt[0])

    group = tabs[6:27]
    mini = tabs[:48]

    # vt_group = inputs.vectorize_all(tabs[6:1000005])

    ''' Check if there are six lines of tabs between each '%' row
    '''
    # check_six_rows(tabs)

    ''' Count # of 6-line chunks that do not have uniform line length
    '''
    check_line_len_uniform(tabs)

    ''' Remove time steps where no note is played.  Currently too many blanks
    (i.e. '-' values).
    '''
    # remove_blanks(tabs)

    # Attempt to flag uneven line lengths (within 6-string groupings) via pandas
    df = pd.DataFrame({'text': tabs})
    df = df[6:]
    df.reset_index(drop=True, inplace=True)
    df['length'] = df.text.apply(lambda x: len(x))
    df.length = df.length * 1.
    df['num'] = df.index + 1.

    grp_column = []
    for group in xrange(len(df)/7):
        for i in xrange(7):
            grp_column.append(group)

    df['chunk'] = grp_column
    # df.loc[df.text=='%\n', 'chunk'] = -1
    # df_check = (df.groupby('chunk')['length'].mean() * 7. - 2) / 6
    df_check = (df.groupby('chunk')['length'].sum() - 2)
    df_check = df_check.to_frame()
    df_check.rename(columns = {'length': 'grp_avg'}, inplace=True)
    df_check.reset_index(inplace=True)
    df_merged = df.merge(df_check, left_on='chunk', right_on='chunk', how='left')
    df_merged['divider'] = df_merged.text == '%\n'


    # temp, clean, removed = remove_uneven_chunks(tabs[6:100001])
