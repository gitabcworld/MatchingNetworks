
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import unittest
import numpy.testing as npt
import torch.nn.functional as F

class DistanceNetwork(nn.Module):
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):

        """
        Produces pdfs over the support set classes for the target set image.
        :param support_set: The embeddings of the support set images, tensor of shape [sequence_length, batch_size, 64]
        :param input_image: The embedding of the target image, tensor of shape [batch_size, 64]
        :return: Softmax pdf
        """
        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            # TODO: Check this variable clipped. Backwards problem??
            sum_support_clipped = Variable(torch.from_numpy(sum_support.data.numpy().clip(eps, float("inf"))), requires_grad=True)
            support_magnitude = torch.rsqrt(sum_support_clipped)
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_magnitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities

class DistanceNetworkTest(unittest.TestCase):
    def setUp(self):

        self.support_set = np.load('/home/aberenguel/pytorch/examples/MatchingNetworks/data/DistanceNetwork/support_set.npy')
        self.input_image = np.load('/home/aberenguel/pytorch/examples/MatchingNetworks/data/DistanceNetwork/input_image.npy')
        self.similarities = np.load('/home/aberenguel/pytorch/examples/MatchingNetworks/data/DistanceNetwork/similarities.npy')

    def tearDown(self):
        pass

    def test_forward(self):
        distNet = DistanceNetwork()
        support_set = Variable(torch.from_numpy(self.support_set), requires_grad=True)
        input_image = Variable(torch.from_numpy(self.input_image), requires_grad=True)
        similarities = distNet(support_set,input_image)
        try:
            npt.assert_array_almost_equal(similarities.data.numpy(), self.similarities.transpose())
        except ZeroDivisionError:
            print "Preds are not equal"


if __name__ == '__main__':
    unittest.main()