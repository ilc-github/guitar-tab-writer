import numpy as np
import pandas as pd
from collections import Counter
import time
import re


''' Modify starter project data (500mb guitar tabs)
 * create smaller dataset for experimentation (~300k lines)
 * create pre-processed input file (alternate input layout; reduce character
 class imbalances)
'''


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

    def vectorize_one_chunk_numpy(self, list_text):
        '''
        INPUT:  list containing six lines of text (ordered as E, B, G, D, A, E
                strings)
        OUTPUT: list of lists, each row representing a time-step (column of single
                tab values, one for each of six strings E-B-G-D-A-E); columns
                represent one-hot according to char_indices
        '''
        steps = len(list_text[0]) - 1
        np_dim = (steps, len(self.chars) * 6)
        vt = np.zeros(np_dim)
        for string_num in xrange(6):
            for i, char in enumerate(list_text[string_num][1:]):
                col_index_bump = (string_num) * len(self.chars)
                vt[i, col_index_bump + self.char_indices[char]] = 1
        return vt

    def vec_one_chunk_pandas(self, str_block):
        '''
        INPUT:  list containing six lines of text (ordered as E, B, G, D, A, E
                strings)
        OUTPUT: list of lists, each row representing a time-step (column of single
                tab values, one for each of six strings E-B-G-D-A-E); columns
                represent one-hot according to char_indices
        '''
        len_row = str_block.find('\n',1)
        steps = len(str_block[2:len_row]) + 1 # Ignore initial '\n' & string name
        np_dim = (steps, len(self.chars) * 6)
        vt = np.zeros(np_dim)
        for string_num in xrange(6):
            lower = str_block.find('\n', string_num * len_row) + 2
            upper = str_block.find('\n', string_num * len_row + 1) + 1
            for i, char in enumerate(str_block[lower:upper]):
                col_index_bump = (string_num) * len(self.chars)
                vt[i, col_index_bump + self.char_indices[char]] = 1
        return vt

def vec_from_file(filename):
    blocks = read_blocks(filename)
    blocks = remove_return(blocks)
    df = pd.DataFrame({'blocks': blocks})
    df.iloc[0] = '\n' + df.iloc[0]
    df = df.iloc[:-1]

    check = df.iloc[0]['blocks']

    df['block_onehot'] = df['blocks'].map(lambda x: inputs.vec_one_chunk_pandas(x))
    vector_tabs = np.concatenate(df.block_onehot.values, axis=0)
    return vector_tabs

def flatten_from_file(filename):
    blocks = read_blocks(filename)
    blocks = remove_return(blocks)
    df = pd.DataFrame({'blocks': blocks})
    df.iloc[0] = '\n' + df.iloc[0]
    df = df.iloc[:-1]

    df['flat'] = df['blocks'].map(lambda x: flatten_text(x))
    list_text = list(df[df['flat'] != '']['flat'])
    return list_text

def remove_return(list_text):
    if list_text[0].find('\r') != -1:
        list_clean = []
        for line in list_text:
            list_clean.append(line.replace('\r', ''))
    return list_clean


def check_line_len_uniform(list_text, verbose=False): # SLOW, procedural list format
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
                    if verbose:
                        print list_text[i] # checking for uneven lines
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


def remove_uneven_chunks(list_text):
    '''
    Check if line length is uniform for each set of 6 tab lines (each set
    separated by '%').  Removes tab chunks that have uneven line lengths.

    INPUT: list of text, lines of guitar tabs
    OUTPUT: list of text, lines of guitar tabs (cleaned)
    '''
    ''' Flag uneven line lengths (within 6-string groupings) via pandas
    '''
    list_text = remove_return(list_text)
    df = pd.DataFrame({'text': list_text})
    df.reset_index(drop=True, inplace=True)
    df['length'] = df.text.map(lambda x: len(x))
    df.length = df.length * 1.
    df['num'] = df.index + 1.

    grp_column = []
    for group in xrange(len(df)/7):
        for i in xrange(7):
            grp_column.append(group)

    df['chunk'] = grp_column # add tab 'chunk' group ID column
    df_temp = (df.groupby('chunk')['length'].sum() - 2).to_frame() # use grp sum
    df_temp.rename(columns = {'length': 'grp_sum'}, inplace=True)
    df_temp.reset_index(inplace=True)
    df_merged = df.merge(df_temp, left_on='chunk', right_on='chunk', how='left')
    df_merged['divider'] = df_merged.text == '%\n'
    df_merged['check'] = df_merged.grp_sum % df.length
    df_merged['drop_flag'] = 0 # mark tab 'chunks' that need to be dropped
    df_merged.loc[df_merged.check != 0, 'drop_flag'] = 1 # flag uneven rows
    df_merged2 = df_merged[df_merged.divider == False] \
                                .groupby('chunk')['drop_flag'].max() # flag grps
    df_merged2 = df_merged2.to_frame()
    df_merged2.reset_index(inplace=True)

    df_new = df.merge(df_merged2, left_on='chunk', right_on='chunk', how='left')
    df_new = df_new[df_new.drop_flag == 0]
    return list(df_new.text)


def remove_blanks(list_text): # remove time steps where no notes are playing (reduce number of '-' throughout tab inputs)
    '''
    Remove time steps where no notes are playing.  Reduce number of '-'
    throughout tab inputs--produce less sparse input.
    INPUT: list of text, lines of guitar tabs
    OUTPUT: list of text, lines of guitar tabs
    '''
    pass

def flatten_text(str_block):
    '''
    Rearrange input tab text from the following format:

    'E|------|\n'
    'B|--1---|\n'
    'G|----2-|\n'
    'D|-0----|\n'
    'A|------|\n'
    'E|------|\n'

    ...to a flat text format:

    '||||||.------.---0--.-1----.------.--2---.------.||||||.\n\n\n\n\n\n'

    INPUT: single string of guitar tabs (6-line chunk)
    OUTPUT: single string, flattened text
    '''
    len_row = str_block.find('\n',1)
    lines = []
    for string_num in xrange(6):
        lower = str_block.find('\n', string_num * len_row) + 2
        upper = str_block.find('\n', string_num * len_row + 1) + 1
        lines.append(str_block[lower:upper])
    flat_txt = ''
    try:
        for i in xrange(len_row - 1):
            for string_num in xrange(6):
                flat_txt += lines[string_num][i]
            flat_txt += '.'
        return flat_txt
    except IndexError:
        return ''


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

def write_to_txt(list_text, filename):
    with open('../data/' + filename + '.txt', 'w') as f:
        for line in list_text:
            f.write(line)

def read_blocks(filename):
    with open(filename) as f:
        txt = f.read()
    blocks = re.split('%', txt)
    return blocks


if __name__ == '__main__':

    ''' Clean inputs.txt
    * Remove tab clusters with uneven line lengths (within 6-string groupings)
    * Write cleaned input to 'input_clean.txt'
    '''
    inputs = ProcessInputs('../data/input.txt')
    # tabs = inputs.gen_list()
    # tabs_clean = remove_uneven_chunks(tabs[6:]) # 1st grp missing line
    # write_to_txt(tabs_clean, 'input_clean')

    # inputs = ProcessInputs('../data/skyrim_repeat.txt')
    # tabs = inputs.gen_list()

    ''' Check if there are six lines of tabs between each '%' row
    '''
    # check_six_rows(tabs)

    ''' Count # of 6-line chunks that do not have uniform line length
    '''
    # check_line_len_uniform(tabs)

    ''' df_merged2 seeing 829073 chunks, 27061 need to be removed

    In [25]: check_line_len_uniform(tabs)
    Number of tab chunks with uneven line lengths =  27071
    Number of tab chunks with even line lengths =  802003
    % uneven =  0.0326520913694
    '''

    '''
    Create smaller cleaned input files
    '''
    # for i in xrange(18):
    #     with open('../data/clean_mini_input' + str(i+1) + '.txt', 'w') as f:
    #         for item in tabs[299999 * i:299999 * (i + 1)]:
    #           f.write("%s" % item)
    # i = 18
    # with open('../data/clean_mini_input' + str(i+1) + '.txt', 'w') as f:
    #     for item in tabs[299999 * i:]:
    #       f.write("%s" % item)

    '''
    Save 2D numpy vectorized one-hots for cleaned input files

    Data formatting issue present for the following files:
    * clean_mini_input4.txt
    * clean_mini_input10.txt
    * clean_mini_input11.txt
    * clean_mini_input.txt
    '''
    # tabs_vector = vec_from_file('../data/clean_mini_input14.txt')
    # np.save('../data/clean14_np_mat.npy', tabs_vector)

    # tabs_vector.dump('tabs_vector.dat')


    ''' Remove time steps where no note is played.  Currently too many blanks
    (i.e. '-' values).
    '''
    # remove_blanks(tabs)

    '''
    Transform one block into alt text layout (flat text)
    Write to text file
    '''
    flat_list = flatten_from_file('../data/input_clean.txt')
    write_to_txt(flat_list, 'flat_tabs_clean')

    temp = None
    with open('../data/flat_tabs_clean.txt') as f:
        # for line in f:
        #     temp.append(line)
        temp = f.read()
    with open('../data/flat_tabs_mini1.txt', 'w') as f:
        f.write(temp[:70000364])
