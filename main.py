##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datasets import omniglotNShot
from option import Options

#Dummy test
import torch
from torch.autograd import Variable
import torch.nn as nn

#args = Options().parse()
#a = omniglotNShot.OmniglotNShotDataset()

from models.BidirectionalLSTM import BidirectionalLSTM

'''
#Function to make dummy data.
def datagen(batch_size, seq_length,  vector_dim):
    return torch.rand(seq_length, batch_size,  vector_dim)

samples = 100000
batch_size = 32
sequence_len = 20
vector_dim = 64
layer_sizes = [100, 100, 100]

lstm = BidirectionalLSTM(layer_sizes = layer_sizes, batch_size = batch_size, vector_dim = 64).cuda()

for sample in range(samples):

    input = Variable(datagen(batch_size, sequence_len, vector_dim).cuda(), requires_grad = True)
    hidden, output = lstm(input)


b = 0
'''


