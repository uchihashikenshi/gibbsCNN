import sys
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from PIL import Image
import cPickle as pickle
cuda.init()

parentpath= '/home/koyama-m/Research/membrane_CNN/'
models_path='/home/koyama-m/Research/membrane_CNN/models/'

sys.path.append(parentpath)
sys.path.append(models_path)

reconstruction_path = '/home/koyama-m/Research/membrane_CNN/data/reconstructed_256images_crop15'
probmap_prefix = 'multi_crop_prediction_image_256_'
binmap_prefix = 'multi_crop_prediction_binary_image_256_'


class gibbs(object):

    def __init__(self, model, patchsize):
        self.model = model
        self.patchsize = patchsize
        pass

    def getfile(self, slice_index):

        filename = 'edgemap%s.tif' % slice_index
        filename2 = 'pooled_image_%s.tif'% slice_index
        filename3 = 'multi_crop_prediction_binary_image_256_%s.tif'% slice_index

        label_data_dir  = 'raw/train-labels'
        raw256path ="preprocessed/training/pooled_training_dataset"
        crop15init_image_path = "reconstructed_256images_crop15"

        testimage=  Image.open("%s/data/%s/%s" % (parentpath, raw256path, filename2))
        testlabel =Image.open("%s/data/%s/%s" % (parentpath, label_data_dir, filename))
        test_reconstruct = Image.open("%sdata/%s/%s" % (parentpath, crop15init_image_path, filename3))

        #CNN Cropped the images. I must register the image with the cropped version.
        crop_cornerX = self.patchsize/2
        crop_cornerY = self.patchsize/2
        crop_range = (crop_cornerX, crop_cornerY, 256-crop_cornerX-1,  256-crop_cornerX-1)

        testimage= testimage.crop(crop_range)
        testlabel= testlabel.crop(crop_range)


        #CONVERT FILE to READABLE FORM
        testimage_array = np.array(list(testimage.getdata()))
        test_reconstruct_array = np.array(list(test_reconstruct.getdata()))
        label_array = np.array(list(testlabel.getdata()))

        reconstr_array = test_reconstruct_array.copy().reshape(np.sqrt(testimage_array.shape[0]), np.sqrt(testimage_array.shape[0]))/255.
        raw_2darray = testimage_array.reshape(np.sqrt(testimage_array.shape[0]), np.sqrt(testimage_array.shape[0]))/255.

        return raw_2darray, reconstr_array, label_array

    def run(self, num_iter, init_heat, cooldown, threshold, n_gibbs_batch, reconstr_array, label_array, raw_2darray):


        heatnow= init_heat
        for iter in range(0, num_iter):
            cornerX = np.random.choice(np.array(range(0,reconstr_array.shape[0]-15)), n_gibbs_batch)
            cornerY=  np.random.choice(np.array(range(0,reconstr_array.shape[0]-15)), n_gibbs_batch)
            #Coordinates subject to change
            coords = np.concatenate((cornerX.reshape(n_gibbs_batch,1), cornerY.reshape(n_gibbs_batch,1)), axis = 1)
            localcenter = np.ones([n_gibbs_batch, 2], int)*7
            centers = coords + localcenter
            centers[0], coords[0]

            gibb_batch_x = np.zeros([n_gibbs_batch, 2, 15, 15],'float32')
            for k in range(0, n_gibbs_batch):
                    corner_x, corner_y = coords[k]
                    yrange = np.array(range(corner_y, corner_y + self.patchsize))
                    xrange = np.array(range(corner_x, corner_x + self.patchsize))
                    patch_raw = raw_2darray[:, yrange][xrange, :]
                    patch_label = reconstr_array[:, yrange][xrange, :]
                    gibb_batch_x[k,0, :,:] = patch_raw
                    gibb_batch_x[k,1, :,:] = patch_label

            gibb_batch_y = np.array([1]*n_gibbs_batch,dtype='int32')
            nll,accuracy,pred= self.model.forward(gibb_batch_x, gibb_batch_y)
            print "iteration : %s,  heat : %s" %(str(iter), str(heatnow))

            heat_pred = np.power(pred.data[:,0], heatnow)
            for k in range(0, n_gibbs_batch):
                if heat_pred[k] > threshold:
                    newlabel = 0.
                else:
                    newlabel = 1.
                reconstr_array[centers[k][0]][centers[k][1]]  = newlabel
            heatnow = heatnow * cooldown



        return reconstr_array
