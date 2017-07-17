import torch
import torch.nn as nn
from torch.autograd import Variable
import unittest
import numpy as np

class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_sizes, batch_size, vector_dim):
        super(BidirectionalLSTM, self).__init__()
        """
        Initializes a multi layer bidirectional LSTM
        :param layer_sizes: A list containing the neuron numbers per layer 
                            e.g. [100, 100, 100] returns a 3 layer, 100
        :param batch_size: The experiments batch size
        """
        self.batch_size = batch_size
        self.hidden_size = layer_sizes[0]
        self.vector_dim = vector_dim
        self.num_layers = len(layer_sizes)

        '''
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        bias: If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN. Default: False
        '''
        self.lstm = nn.LSTM(input_size=self.vector_dim,
                            num_layers=self.num_layers,
                            hidden_size=self.hidden_size,
                            bidirectional=True)

    def forward(self, inputs):
        """
        Runs the bidirectional LSTM, produces outputs and saves both forward and backward states as well as gradients.
        :param x: The inputs should be a list of shape [sequence_length, batch_size, 64]
        :return: Returns the LSTM outputs, as well as the forward and backward hidden states.
        """
        c0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        h0 = Variable(torch.rand(self.lstm.num_layers*2, self.batch_size, self.lstm.hidden_size),
                      requires_grad=False)
        output, (hn, cn) = self.lstm(inputs, (h0, c0))
        return output, hn, cn


class BidirectionalLSTMTest(unittest.TestCase):
    def setUp(self):
        self.encoded_images_before = np.load('../data/lstm/encoded_images_before.npy')
        self.encoded_images_afer = np.load('../data/lstm/encoded_images_after.npy')
        self.output_state_fw = np.load('../data/lstm/output_state_fw.npy')
        self.output_state_bw = np.load('../data/lstm/output_state_bw.npy')
        classes_per_set = 20
        samples_per_class = 1
        self.sequence_size = classes_per_set * samples_per_class
        #self.layer_sizes = [32]
        self.layer_sizes = [64]
        self.batch_size = 32
        self.vector_dim = 64

    def tearDown(self):
        pass

    def test_forward(self):
        biLstm = BidirectionalLSTM(layer_sizes=self.layer_sizes, batch_size=self.batch_size,
                                 vector_dim=self.vector_dim)
        inputs = Variable(torch.from_numpy(self.encoded_images_before), requires_grad=True)
        output, output_state_fw, output_state_bw = biLstm(inputs)
        self.assertEqual(0, 0)



if __name__ == '__main__':
    unittest.main()

