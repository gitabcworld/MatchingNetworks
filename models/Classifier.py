import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import unittest
import numpy as np

def convLayer(in_planes, out_planes, stride=1, padding = 1, bias = True):
    "3x3 convolution with padding"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0)
    )


class Classifier(nn.Module):
    def __init__(self, layer_sizes, num_channels = 1, keep_prob = 0.5):
        super(Classifier, self).__init__()

        """
        Builds a CNN to produce embeddings
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        assert len(layer_sizes)==4, "layer_sizes should be a list of length 4"

        self.layer1 = convLayer(num_channels, layer_sizes[0])
        self.layer2 = convLayer(layer_sizes[0], layer_sizes[1])
        self.layer3 = convLayer(layer_sizes[1], layer_sizes[2])
        self.layer4 = convLayer(layer_sizes[2], layer_sizes[3])


        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

    def weights_init(self,module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.squeeze(x)
        return x


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward(self):
        pass

if __name__ == '__main__':
    unittest.main()
