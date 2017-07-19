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
import torch.nn as nn
from torch.autograd import Variable
from experiments.OneShotBuilder import OneShotBuilder

# Experiment Setup
batch_size = 32
fce = True
classes_per_set = 20
samples_per_class = 1
channels = 1
epochs = 200
logs_path = "one_shot_outputs/"
experiment_name = "one_shot_learning_embedding_{}_{}".format(samples_per_class, classes_per_set)

total_epochs = 300
total_train_batches = 1000
total_val_batches = 100
total_test_batches = 250

args = Options().parse()
data = omniglotNShot.OmniglotNShotDataset(dataroot=args.dataroot, batch_size = batch_size,
                                          classes_per_set=classes_per_set,
                                          samples_per_class=samples_per_class)
obj_oneShotBuilder = OneShotBuilder(data)
obj_oneShotBuilder.build_experiment(batch_size, classes_per_set, samples_per_class, channels, fce)
obj_oneShotBuilder.run_training_epoch(total_train_batches)

