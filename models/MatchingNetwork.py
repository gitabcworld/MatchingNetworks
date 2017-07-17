import torch
import torch.nn as nn
from torch.autograd import Variable
import unittest
import numpy as np
from BidirectionalLSTM import BidirectionalLSTM
from Classifier import Classifier
from DistanceNetwork import DistanceNetwork
from AttentionalClassify import AttentionalClassify

class MatchingNetwork(nn.Module):
    def __init__(self, keep_prob, \
                 batch_size=100, num_channels=1, is_training=False, learning_rate=0.001, rotate_flag=False, fce=False, num_classes_per_set=5, \
                 num_samples_per_class=1):
        super(MatchingNetwork, self).__init__()

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        self.batch_size = batch_size
        self.fce = fce
        self.g = Classifier(layer_sizes=[64, 64, 64 ,64], num_channels=num_channels, )
        if fce:
            self.lstm = BidirectionalLSTM(layer_sizes=[32], batch_size=self.batch_size, vector_dim = 64)
        self.dn = DistanceNetwork()
        self.classify = AttentionalClassify()
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.k = None
        self.rotate_flag = rotate_flag
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_class = num_samples_per_class
        self.learning_rate = learning_rate

    def forward(self, support_set_images, support_set_labels, target_image, target_label):
        """
        Builds graph for Matching Networks, produces losses and summary statistics.
        :param support_set_images: A tensor containing the support set images [sequence_size, batch_size, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [sequence_size, batch_size, 1]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, 28, 28, 1]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :return: 
        """
        # produce embeddings for support set images
        encoded_images = []
        for image in support_set_images:
            gen_encode = self.g(image)
            encoded_images.append(gen_encode)

        # produce embeddings for target images
        gen_encode = self.g(target_image)
        encoded_images.append(gen_encode)
        outputs = torch.stack(encoded_images)

        if self.fce:
            outputs, hn, cn = self.lstm(outputs)

        # get similarity between support set embeddings and target
        similarities = self.dn(support_set=outputs[:-1], input_image=outputs[-1])

        # produce predictions for target probabilities
        preds = self.classify(similarities,support_set_y=support_set_labels)

        #
        values, indices = preds.max(1)
        values, indices = torch.max(tensor, 0)
        values, indices = tensor.max(preds,1)
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.cast(self.target_label, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        crossentropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_label,
                                                                                          logits=preds))

        a = 0


class MatchingNetworkTest(unittest.TestCase):
    def setUp(self):

        self.batch_size = 32
        self.fce = True
        self.classes_per_set = 20
        self.samples_per_class = 1
        self.channels = 1

        self.training_phase = Variable(torch.ByteTensor(1), requires_grad=True)
        self.rotate_flag = Variable(torch.ByteTensor(1), requires_grad=True)
        self.keep_prob = Variable(torch.FloatTensor(1), requires_grad=True)
        self.current_learning_rate = 1e-03
        self.learning_rate = Variable(torch.FloatTensor(1), requires_grad=True)
        self.matchingNet = MatchingNetwork(batch_size=self.batch_size,
                                                 keep_prob=self.keep_prob, num_channels=self.channels,
                                                 is_training=self.training_phase, fce=self.fce, rotate_flag=self.rotate_flag,
                                                 num_classes_per_set=self.classes_per_set,
                                                 num_samples_per_class=self.samples_per_class,
                                                 learning_rate=self.learning_rate)

    def tearDown(self):
        pass

    def test_forward(self):
        sequence_size = self.classes_per_set * self.samples_per_class
        support_set_images = Variable(torch.FloatTensor(sequence_size, self.batch_size, self.channels, 28, 28),
                                           requires_grad=True)
        support_set_labels = Variable(torch.LongTensor(sequence_size, self.batch_size).random_() % self.classes_per_set, requires_grad=True)
        # Create one_hot vectors for support_set_labels
        sequence_length = support_set_labels.size()[0]
        batch_size = support_set_labels.size()[1]
        num_classes = 20
        support_set_labels_onehot = []
        for labels in support_set_labels:
            temp_onehot = torch.FloatTensor(batch_size, num_classes)
            temp_onehot.zero_()
            temp_onehot.scatter_(1, labels.view(-1, 1).data, 1)
            support_set_labels_onehot.append(temp_onehot)
        support_set_labels_onehot = torch.stack(support_set_labels_onehot)
        support_set_labels = Variable(support_set_labels_onehot,requires_grad=True)


        #self.support_set_labels = tf.one_hot(self.support_set_labels, self.num_classes_per_set)  # one hot encode
        target_image = Variable(torch.FloatTensor(self.batch_size, self.channels, 28, 28), requires_grad=True)
        target_label = Variable(torch.LongTensor(self.batch_size).random_() % self.classes_per_set, requires_grad=True)
        self.matchingNet(support_set_images, support_set_labels, target_image, target_label)
        self.assertEqual(0, 0)

if __name__ == '__main__':
    unittest.main()



