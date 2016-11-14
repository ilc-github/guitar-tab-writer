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
