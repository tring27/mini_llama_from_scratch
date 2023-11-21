'''
Utility functions needed for the project
'''
import torch


#return a list of unique characters in a string
def get_vocab(s):
    return sorted(list(set(s)))

#split test and train data
def test_train_split(d, split):
    '''
    d: 1d tensor input
    split: float value to split d into two partitions
    '''
    n = int(split*len(d))
    train = d[:n]
    test = d[n:]
    return train, test

#random sample from the input, and their correspoding labels
#note that the input and label size will be the same here
def get_rand_batch(d, batch_size, context_size, seed):
    #generate a vector of size = batch_size, with random
    #starting points within the data 'd'
    torch.manual_seed(seed)
    ix = torch.randint(0, len(d) - context_size - 1, (batch_size,))
    x = torch.stack([d[i:i+context_size] for i in ix])
    y = torch.stack([d[i+1:i+1+context_size] for i in ix])
    return x, y