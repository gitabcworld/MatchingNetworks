##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datasets import omniglot
import torchvision.transforms as transforms
from PIL import Image
import os.path

import numpy as np
np.random.seed(2191)  # for reproducibility

# LAMBDA FUNCTIONS
filenameToPILImage = lambda x: Image.open(x).convert('L')
PiLImageResize = lambda x: x.resize((28,28))
np_reshape = lambda x: np.reshape(x, (28, 28, 1))

class OmniglotNShotDataset():
    def __init__(self, dataroot, batch_size = 100, classes_per_set=10, samples_per_class=1):

        if not os.path.isfile(os.path.join(dataroot,'data.npy')):
            self.x = omniglot.OMNIGLOT(dataroot, download=True,
                                     transform=transforms.Compose([filenameToPILImage,
                                                                   PiLImageResize,
                                                                   np_reshape]))

            """
            # Convert to the format of AntreasAntoniou. Format [nClasses,nCharacters,28,28,1]
            """
            temp = dict()
            for (img, label) in self.x:
                if label in temp:
                    temp[label].append(img)
                else:
                    temp[label]=[img]
            self.x = [] # Free memory

            for classes in temp.keys():
                self.x.append(np.array(temp[temp.keys()[classes]]))
            self.x = np.array(self.x)
            temp = [] # Free memory
            np.save(os.path.join(dataroot,'data.npy'),self.x)
        else:
            self.x = np.load(os.path.join(dataroot,'data.npy'))

        """
        Constructs an N-Shot omniglot Dataset
        :param batch_size: Experiment batch_size
        :param classes_per_set: Integer indicating the number of classes per set
        :param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
        """

        shuffle_classes = np.arange(self.x.shape[0])
        np.random.shuffle(shuffle_classes)
        self.x = self.x[shuffle_classes]
        self.x_train, self.x_test, self.x_val  = self.x[:1200], self.x[1200:1500], self.x[1500:]
        self.normalization()

        self.batch_size = batch_size
        self.n_classes = self.x.shape[0]
        self.classes_per_set = classes_per_set
        self.samples_per_class = samples_per_class

        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test} #original data cached
        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  #current epoch data cached
                               "val": self.load_data_cache(self.datasets["val"]),
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("train_shape", self.x_train.shape, "test_shape", self.x_test.shape, "val_shape", self.x_val.shape)
        #print("before_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_val = (self.x_val - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std
        #self.mean = np.mean(self.x_train)
        #self.std = np.std(self.x_train)
        #self.max = np.max(self.x_train)
        #self.min = np.min(self.x_train)
        #print("after_normalization", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects 1000 batches data for N-shot learning
        :param data_pack: Data pack to use (any one of train, val, test)
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        n_samples = self.samples_per_class * self.classes_per_set
        data_cache = []
        for sample in range(1000):
            support_set_x = np.zeros((self.batch_size, n_samples, 28, 28, 1))
            support_set_y = np.zeros((self.batch_size, n_samples))
            target_x = np.zeros((self.batch_size, self.samples_per_class, 28, 28, 1), dtype=np.int)
            target_y = np.zeros((self.batch_size, self.samples_per_class), dtype=np.int)
            for i in range(self.batch_size):
                pinds = np.random.permutation(n_samples)
                classes = np.random.choice(data_pack.shape[0], self.classes_per_set, False)
                # select 1-shot or 5-shot classes for test with repetition
                x_hat_class = np.random.choice(classes, self.samples_per_class, True)
                pinds_test = np.random.permutation(self.samples_per_class)
                ind = 0
                ind_test = 0
                for j, cur_class in enumerate(classes):  # each class
                    if cur_class in x_hat_class:
                        # Count number of times this class is inside the meta-test
                        n_test_samples = np.sum(cur_class == x_hat_class)
                        example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class + n_test_samples, False)
                    else:
                        example_inds = np.random.choice(data_pack.shape[1], self.samples_per_class, False)

                    # meta-training
                    for eind in example_inds[:self.samples_per_class]:
                        support_set_x[i, pinds[ind], :, :, :] = data_pack[cur_class][eind]
                        support_set_y[i, pinds[ind]] = j
                        ind = ind + 1
                    # meta-test
                    for eind in example_inds[self.samples_per_class:]:
                        target_x[i, pinds_test[ind_test], :, :, :] = data_pack[cur_class][eind]
                        target_y[i, pinds_test[ind_test]] = j
                        ind_test = ind_test + 1

            data_cache.append([support_set_x, support_set_y, target_x, target_y])
        return data_cache

    def __get_batch(self, dataset_name):
        """
        Gets next batch from the dataset with name.
        :param dataset_name: The name of the dataset (one of "train", "val", "test")
        :return:
        """
        if self.indexes[dataset_name] >= len(self.datasets_cache[dataset_name]):
            self.indexes[dataset_name] = 0
            self.datasets_cache[dataset_name] = self.load_data_cache(self.datasets[dataset_name])
        next_batch = self.datasets_cache[dataset_name][self.indexes[dataset_name]]
        self.indexes[dataset_name] += 1
        x_support_set, y_support_set, x_target, y_target = next_batch
        return x_support_set, y_support_set, x_target, y_target

    def get_batch(self,str_type, rotate_flag = False):

        """
        Get next batch
        :return: Next batch
        """
        x_support_set, y_support_set, x_target, y_target = self.__get_batch(str_type)
        if rotate_flag:
            k = int(np.random.uniform(low=0, high=4))
            # Iterate over the sequence. Extract batches.
            for i in np.arange(x_support_set.shape[0]):
                x_support_set[i,:,:,:,:] = self.__rotate_batch(x_support_set[i,:,:,:,:],k)
            # Rotate all the batch of the target images
            for i in np.arange(x_target.shape[0]):
                x_target[i,:,:,:,:] = self.__rotate_batch(x_target[i,:,:,:,:], k)
        return x_support_set, y_support_set, x_target, y_target


    def __rotate_data(self, image, k):
        """
        Rotates one image by self.k * 90 degrees counter-clockwise
        :param image: Image to rotate
        :return: Rotated Image
        """
        return np.rot90(image, k)


    def __rotate_batch(self, batch_images, k):
        """
        Rotates a whole image batch
        :param batch_images: A batch of images
        :param k: integer degree of rotation counter-clockwise
        :return: The rotated batch of images
        """
        batch_size = len(batch_images)
        for i in np.arange(batch_size):
            batch_images[i] = self.__rotate_data(batch_images[i], k)
        return batch_images
