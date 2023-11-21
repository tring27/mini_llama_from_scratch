import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.nn import functional as F
import time
from matplotlib import pyplot as plt
import utils as ut

#setting seed for all random blocks in the model
seed = 27
torch.manual_seed(seed)

#load input data, which is Tiny Shakespear in our case
with open('./data/input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# print(text[0:1000])
#so far we have the complete text loaded in the variable 'text'

#get the vocab, meaning a list of unique characters in the whole text
vocab = ut.get_vocab(text)

vocab_size = len(vocab)

#models do not understand character, so we need a way to convert characters into numbers
#and humans can't read numbers, and so the model output will need to be converted back to characters
#there are various encoders and decoders that are very exotic in their features
#but we will build something simple and homegrown

#string to integer is just going to assign an int value to every unique character in our vocab
stoi = { c:i for i,c in enumerate(vocab) }

#encode function takes input as string
#returns a list of integer for each character in the string
encode = lambda s: [stoi[c] for c in s] 

#integer to string is just going to do the opposite of stoi
itos = { i:c for i,c in enumerate(vocab) }

#decode function takes a list of integers as input
#returns string of characters
decode = lambda l: ''.join(itos[i] for i in l)

#testing
# print(encode("hello world!"))
# print(decode(encode("hello world!")))

#the encoder and decoder are also known as "tokenizer" and "detokenizer"

#now lets tokenize the entire input dataset and convert to tensor

data = torch.tensor(encode(text), dtype = torch.long)

#lets see how it looks like
# print(data.shape)

#Now lets split our data into train and validation sets

train_data, val_data = ut.test_train_split(data, 0.9)
# print(train_data.shape)
# print(val_data.shape)

#lets get a random batch of inputs (x) and labels (y)
xb, yb = ut.get_rand_batch(train_data, batch_size=4, context_size=8, seed=seed)
# print(xb.shape)
# print(yb.shape)



