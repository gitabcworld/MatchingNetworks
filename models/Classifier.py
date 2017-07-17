import torch
import torch.nn as nn
from torch.autograd import Variable
import unittest
import numpy as np
import math


def convLayer(in_planes, out_planes, stride=1, padding = 1, bias = True):
    "3x3 convolution with padding"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(),
        nn.BatchNorm2d(out_planes),
        # nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout()
    )


class Classifier(nn.Module):
    def __init__(self, layer_sizes, num_channels = 1, keep_prob = 0.5):
        super(Classifier, self).__init__()

        """
        Builds a CNN to produce embeddings
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param num_channels: Number of channels of images
        """
        #self.num_channels = num_channels
        #self.layer_sizes = layer_sizes
        assert len(layer_sizes)==4, "layer_sizes should be a list of length 4"

        self.layer1 = convLayer(num_channels, layer_sizes[0])
        self.layer2 = convLayer(layer_sizes[0], layer_sizes[1])
        self.layer3 = convLayer(layer_sizes[1], layer_sizes[2])
        self.layer4 = convLayer(layer_sizes[2], layer_sizes[3])

        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)
        '''
        # Module initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        '''

    def weights_init(self,m):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()


    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        # TODO: What is better 2 padding at first conv2d or 1 padding in the last layer of conv2d??
        #x = nn.Conv2d(1, 64, kernel_size=3, stride=1, bias=True).cuda()(image_input)
        #x = nn.LeakyReLU().cuda()(x)
        #x = nn.BatchNorm2d(64).cuda()(x)
        #x = nn.MaxPool2d(kernel_size=2, stride=2).cuda()(x)
        #x = nn.Dropout().cuda()(x)

        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.squeeze(x)
        # TODO: flat output
        return x


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.inputs = np.load('../data/target_image.npy')
        self.inputs = np.reshape(self.inputs, (32, 1, 28, 28))
        self.outputs = np.load('../data/gen_encode.npy')
        self.layer_sizes = [64,64,64,64]

    def tearDown(self):
        pass

    def test_forward(self):
        classifier = Classifier(layer_sizes=self.layer_sizes).cuda()
        input = Variable(torch.from_numpy(self.inputs).cuda(), requires_grad=True)
        output = classifier(input)
        # TODO: why the output contains so many 0? The self.outputs d
        print("sum output tf: %f" % np.sum(self.outputs))
        print("sum output pytorch: %f" % np.sum(output.cpu().data.numpy()))

        a = 0
        #self.assertEqual(0, 0)


if __name__ == '__main__':
    unittest.main()
