
list = ['the', 'end', 'of', 'the', 'few', 'door', 'the', 'end', 'of', 'the', 'aside', 'with', 'his', 'free', 'hand.', 'the', 'kitchen', 'door', 'opened']

list2 = ['the', 'end', 'of', 'the', 'few', 'door', 'the', 'end', 'of', 'the', 'aside', 'with', 'his', 'free', 'hand.', 'the', 'kitchen', 'door', 'opened']
list2.pop(0)
print(' '.join(map(str, list)) + ' '.join(map(str, list2)))
