
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import unittest
import numpy.testing as npt

class AttentionalClassify(nn.Module):
    def __init__(self):
        super(AttentionalClassify, self).__init__()

    def forward(self, similarities, support_set_y):

        """
        Produces pdfs over the support set classes for the target set image.
        :param similarities: A tensor with cosine similarities of size [sequence_length, batch_size]
        :param support_set_y: A tensor with the one hot vectors of the targets for each support set image
                                                                            [sequence_length,  batch_size, num_classes]
        :return: Softmax pdf
        """
        softmax = nn.Softmax()
        softmax_similarities = softmax(similarities)
        preds = softmax_similarities.unsqueeze(1).bmm(support_set_y).squeeze()
        #softmax_similarities = nn.Softmax(similarities)
        #preds = tf.squeeze(tf.matmul(tf.expand_dims(softmax_similarities, 1), support_set_y))
        return preds

class AttentionalClassifyTest(unittest.TestCase):
    def setUp(self):
        self.similarities = np.load('../data/similarities.npy')
        self.support_set_y = np.load('../data/support_set_y.npy')
        self.softmax_similarities = np.load('../data/softmax_similarities.npy')
        self.preds = np.load('../data/preds.npy')

    def tearDown(self):
        pass

    def test_forward(self):
        attCls = AttentionalClassify().cuda()
        similarities = Variable(torch.from_numpy(self.similarities).cuda(), requires_grad=True)
        support_set_y = Variable(torch.from_numpy(self.support_set_y).cuda(), requires_grad=True)
        preds = attCls(similarities,support_set_y)
        try:
            npt.assert_array_almost_equal(preds.cpu().data.numpy(), self.preds)
        except ZeroDivisionError:
            print "Preds are not equal"


if __name__ == '__main__':

    unittest.main()