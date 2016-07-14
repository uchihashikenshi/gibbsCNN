import sys
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from matplotlib import pyplot as plt
from PIL import Image
import cPickle as pickle
cuda.init()

holesize = 0
cropsize = 15
coolrate = 0.95
epoch = 80


parentpath= '/home/koyama-m/Research/membrane_CNN/'
models_path='/home/koyama-m/Research/membrane_CNN/models/'

sys.path.append(parentpath)
sys.path.append(models_path)

reconstruction_path = '/home/koyama-m/Research/membrane_CNN/data/reconstructed_256images_crop%s'%str(cropsize)
probmap_prefix = 'multi_crop_prediction_image_256_'
binmap_prefix = 'multi_crop_prediction_binary_image_256_'

#LOAD MODEL
modelname = 'hole%s_cool_rate%sconditional_distr_trained_model256_crop%sepoch%s.pkl'%(str(holesize),str(coolrate), str(cropsize),str(epoch))
print models_path +modelname

#NOT DONE....