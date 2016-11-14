''' EDA scripts
 * character frequency
'''

from collections import Counter

path = '../data/input.txt'

with open(path, 'r') as f:
    ans = []
    for line in f:
        ans.append(line)

# character frequency
