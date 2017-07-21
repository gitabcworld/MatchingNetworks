import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import unittest
import numpy as np
import math


def convLayer(in_planes, out_planes, stride=1, padding = 1, bias = True):
    "3x3 convolution with padding"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=padding, bias=bias),
        #nn.LeakyReLU(0.2),
        nn.LeakyReLU(),
        nn.BatchNorm2d(out_planes),
        #nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.1)
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
        self.weights_init_tensorflow(self.layer1,1)
        self.weights_init_tensorflow(self.layer2,2)
        self.weights_init_tensorflow(self.layer3,3)
        self.weights_init_tensorflow(self.layer4,4)
        '''

    def weights_init(self,module):
        #for m in self.modules():
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2))
                init.constant(m.bias, 0)
                ##m.weight.data.normal_(0.0, 0.02)
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.normal_(1.0, 0.02)
                #m.bias.data.fill_(0)
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def weights_init_tensorflow(self,module,layer):
        #for m in self.modules():
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = torch.from_numpy(np.load('/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/conv' + str(layer) + '_weigths.npy').transpose((3,2,1,0)))
                m.weight.data = m.weight.data.contiguous()
                m.bias.data = torch.from_numpy(np.load('/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/conv' + str(layer) + '_bias.npy'))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = torch.from_numpy(np.load(
                    '/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/bn' + str(
                        layer) + '_gamma.npy'))
                m.bias.data = torch.from_numpy(np.load(
                    '/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/bn' + str(
                        layer) + '_beta.npy'))


    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """

        #x = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=True).cuda()(image_input)
        #x = nn.LeakyReLU().cuda()(x)
        #x = nn.BatchNorm2d(64).cuda()(x)
        #x = nn.MaxPool2d(kernel_size=2, stride=2).cuda()(x)
        #x = nn.Dropout().cuda()(x)

        #check = np.sum(self.layer1[0](image_input).data.cpu().numpy())

        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.squeeze(x)
        return x


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.inputs = np.load('../data/target_image.npy')
        self.inputs = np.reshape(self.inputs, (32, 1, 28, 28))
        self.outputs = np.load('../data/gen_encode.npy')
        self.layer_sizes = [64,64,64,64]

        self.after_conv1 = np.load(
            '/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/after_conv1_encoder.npy')
        self.after_conv2 = np.load(
            '/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/after_conv2_encoder.npy')
        self.after_conv3 = np.load(
            '/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/after_conv3_encoder.npy')
        self.after_conv4 = np.load(
            '/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/after_conv4_encoder.npy')
        self.image_input = np.load(
            '/home/aberenguel/pytorch/examples/MatchingNetworks/data/classifier/image_input.npy')

    def tearDown(self):
        pass

    def test_forward(self):
        classifier = Classifier(layer_sizes=self.layer_sizes).cuda()

        print("sum conv1 tf: %f" % np.sum(self.after_conv1))
        print("sum conv2 tf: %f" % np.sum(self.after_conv2))
        print("sum conv3 tf: %f" % np.sum(self.after_conv3))
        print("sum conv4 tf: %f" % np.sum(self.after_conv4))
        input = Variable(torch.from_numpy(self.image_input.transpose((0,3,1,2))).cuda(), requires_grad=True)

        x1 = classifier.layer1(input)
        print("sum conv1 pytorch: %f" % torch.sum(x1).data[0])
        x2 = classifier.layer2(x1)
        print("sum conv2 pytorch: %f" % torch.sum(x2).data[0])
        x3 = classifier.layer3(x2)
        print("sum conv2 pytorch: %f" % torch.sum(x3).data[0])
        x4 = classifier.layer4(x3)
        print("sum conv2 pytorch: %f" % torch.sum(x4).data[0])

        input = Variable(torch.from_numpy(self.inputs).cuda(), requires_grad=True)
        output = classifier(input)
        # TODO: why the output contains so many 0? The self.outputs d
        print("sum output tf: %f" % np.sum(self.outputs))
        print("sum output pytorch: %f" % np.sum(output.cpu().data.numpy()))

        a = 0
        #self.assertEqual(0, 0)


if __name__ == '__main__':
    unittest.main()
