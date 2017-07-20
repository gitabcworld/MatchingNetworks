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
import tqdm

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

best_val = 0.
with tqdm.tqdm(total=total_epochs) as pbar_e:
    for e in range(0, total_epochs):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch(total_train_batches=total_train_batches)
        print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

        total_val_c_loss, total_val_accuracy = obj_oneShotBuilder.run_validation_epoch(
            total_val_batches=total_val_batches)
        print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

        if total_val_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
            best_val = total_val_accuracy
            total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch(
                total_test_batches=total_test_batches)
            print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
        else:
            total_test_c_loss = -1
            total_test_accuracy = -1

        pbar_e.update(1)
