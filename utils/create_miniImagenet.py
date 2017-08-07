##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
This code creates the MiniImagenet dataset. Following the partitions given
by Sachin Ravi and Hugo Larochelle in 
https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet
'''

import numpy as np
import csv
import glob, os
from shutil import copyfile
from tqdm import tqdm

pathImageNet = '/home/aberenguel/Dataset/Imagenet/ILSVRC2012_img_train'
files = ['../data/miniImagenet/train.csv','../data/miniImagenet/val.csv','../data/miniImagenet/test.csv']

pathImages = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          '..','data/miniImagenet/images/')

# Check if the folder of images exist. If not create it.
if not os.path.exists(pathImages):
    os.makedirs(pathImages)

for filename in files:
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader, None)
        images = {}
        print('Reading IDs....')
        for row in tqdm(csv_reader):
            if row[1] in images.keys():
                images[row[1]].append(row[0])
            else:
                images[row[1]] = [row[0]]

        print('Writing photos....')
        for c in tqdm(images.keys()): # Iterate over all the classes
            os.chdir(pathImageNet) # TODO: change this line that is change the current folder.
            lst_files = []
            for file in glob.glob("*"+c+"*"):
                lst_files.append(file)
            # TODO: Sort by name of by index number of the image???
            # I sort by the number of the image
            lst_index = [int(i[i.index('_')+1:i.index('.')]) for i in lst_files]
            index_sorted = sorted(range(len(lst_index)), key=lst_index.__getitem__)

            # Now iterate
            index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[c]]
            selected_images = np.array(index_sorted)[np.array(index_selected) - 1]
            for i in np.arange(len(selected_images)):
                copyfile(os.path.join(pathImageNet,lst_files[selected_images[i]]),
                         os.path.join(pathImages, images[c][i]))



