import sys
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from matplotlib import pyplot as plt
from scipy import misc
from PIL import Image
import cPickle as pickle


class Generate_laplace_pyramid(object):

    def __init__(self, patchsize_for_rcstr_data = 15,  resolution = 256, numslice = 100):
        self.patchsize_for_rcstr_data = patchsize_for_rcstr_data
        self.resolution =256
        self.numslice = 100
        pass

    def run(self):

        #TRAINING
        # (1) Save shifted raw files   at ~/Research/membrane_CNN/data/preprocessed/training_laplace
        # (2) Save delta files         at ~/Research/membrane_CNN/data/preprocessed/training_laplace

        #Testing
        # (3) Save shifted label files at  ~/Research/membrane_CNN/data/preprocessed/training_laplace

        parent_path = '/home/koyama-m/Research/membrane_CNN/'
        original_raw_filepath = parent_path + "data/preprocessed/training/pooled_training_dataset/"
        original_label_filepath = 'raw/train-labels'
        reconstr_image_path = "reconstructed_256images_crop%s/" %self.patchsize_for_rcstr_data

        save_files_path = parent_path + "data/preprocessed/training_laplace/"


        crop_cornerX = self.patchsize_for_rcstr_data/2
        crop_cornerY = self.patchsize_for_rcstr_data/2
        crop_range = (crop_cornerX, crop_cornerY, self.resolution-crop_cornerX-1,  self.resolution-crop_cornerX-1)

        for slice_index in range(1,self.numslice+1):

            filename_original_raw = "pooled_image_%03d.tif" %slice_index
            filename_original_label = 'edgemap%03d.tif' %slice_index
            reconstr_label_filename = 'multi_crop_prediction_binary_image_%s_%03d.tif'%(self.resolution, slice_index)
            #reconstr_prob_filename = 'multi_crop_prediction_image_%s_%03d.tif'%(self.resolution, slice_index)

            raw_original = Image.open(original_raw_filepath + filename_original_raw)
            real_label_original = Image.open("%s/data/%s/%s" % (parent_path, original_label_filepath, filename_original_label ))
            reconstr_label_original= Image.open("%sdata/%s/%s" % (parent_path, reconstr_image_path, reconstr_label_filename))
            #reconstr_prob_original= Image.open("%sdata/%s/%s" % (parent_path, reconstr_image_path, reconstr_prob_filename))

            raw_image = raw_original.crop(crop_range)
            real_label = real_label_original.crop(crop_range)
            reconstr_label = reconstr_label_original

            reshape_dimen= self.resolution-self.patchsize_for_rcstr_data
            real_label_array = np.array(list(real_label.getdata())).reshape(reshape_dimen,reshape_dimen)/255.
            recontsr_label_array = np.array(list(reconstr_label.getdata())).reshape(reshape_dimen,reshape_dimen)/255.

            delta = real_label_array - recontsr_label_array

            raw_filename = "pooled_image_%03d_margin%s.tif" %(slice_index, self.patchsize_for_rcstr_data)
            label_filename = "edgemap%03d_margin%s.tif"%(slice_index, self.patchsize_for_rcstr_data)
            delta_filename = "deltamap%03d_margin%s.tif"%(slice_index, self.patchsize_for_rcstr_data)

            misc.imsave(save_files_path + raw_filename , raw_image)
            misc.imsave(save_files_path + label_filename , real_label)
            misc.imsave(save_files_path + delta_filename , delta)





