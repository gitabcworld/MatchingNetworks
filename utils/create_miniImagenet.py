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
#from shutil import copyfile
import cv2
from tqdm import tqdm

pathImageNet = '/home/aberenguel/Dataset/Imagenet/ILSVRC2012_img_train'
pathminiImageNet = '/home/aberenguel/Dataset/miniImagenet/'
pathImages = os.path.join(pathminiImageNet,'images/')
filesCSVSachinRavi = [os.path.join(pathminiImageNet,'train.csv'),
                      os.path.join(pathminiImageNet,'val.csv'),
                      os.path.join(pathminiImageNet,'test.csv')]

# Check if the folder of images exist. If not create it.
if not os.path.exists(pathImages):
    os.makedirs(pathImages)

for filename in filesCSVSachinRavi:
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
            #os.chdir(pathImageNet) # TODO: change this line that is change the current folder.
            lst_files = []
            for file in glob.glob(pathImageNet + "/*"+c+"*"):
                lst_files.append(file)
            # TODO: Sort by name of by index number of the image???
            # I sort by the number of the image
            lst_index = [int(i[i.rfind('_')+1:i.rfind('.')]) for i in lst_files]
            index_sorted = sorted(range(len(lst_index)), key=lst_index.__getitem__)

            # Now iterate
            index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[c]]
            selected_images = np.array(index_sorted)[np.array(index_selected) - 1]
            for i in np.arange(len(selected_images)):
                # read file and resize to 84x84x3
                im = cv2.imread(os.path.join(pathImageNet,lst_files[selected_images[i]]))
                im_resized = cv2.resize(im, (84, 84), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(pathImages, images[c][i]),im_resized)

                #copyfile(os.path.join(pathImageNet,lst_files[selected_images[i]]),os.path.join(pathImages, images[c][i]))




