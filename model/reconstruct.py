#######################
##Loading Reconstruction ####
#######################


#Variables
cropsize = 15 
resolution = 256
threshold = 0.52  #cutoff of binary
epochsize = 100

multicrop = True

print 'Loading the required modules... ' 

import os
import pickle
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
from matplotlib import pyplot as plt
from PIL import Image
import load_preprocessed_data_set as ppr
    


print 'Confirming the direcoties...\n ' 
#Choice of the directory to save the reconstructed images
prefix = '/home/koyama-m/Research/membrane_CNN/'
data_path = prefix +'data/'
save_path = data_path + 'reconstructed_%simages_crop%s/'%(str(resolution), str(cropsize))
if os.path.exists(save_path) != True:
    os.mkdir(save_path)
    
models_path = prefix + 'models/'
path_to_store_temporary_pkl = data_path + 'temp_data/'

#Filepath of preprocessed cropped files reeady for reconstructions

print 'loading the preprocessed cropped files reeady for reconstructions'
datafilepath = data_path+'reconstruction_All_%s_dataset_crop%s' %(str(resolution), str(cropsize))
rcstr_path = data_path + "test_dataset/reconstruction_All_%s_dataset_crop%s/" %(str(resolution), str(cropsize))
if os.path.exists(rcstr_path) != True:
    print 'The required stack data folder:'
    print rcstr_path
    required_program = 'preprocess_all_reconstruction_data.py'
    print 'not found. This sequence is not ready to run. You need to execute:'
    print required_program + '. \nAborting the sequence...'
    quit()

#Model to be used 
modelname = 'trained_model%s_crop%sepoch%s.pkl'%(str(resolution), str(cropsize), str(epochsize))
if os.path.exists(models_path +modelname) != True:
    print 'Designated model :' + modelname + ' not found! Aborting the sequence...'
    quit()

cuda.init()
model_rc= pickle.load(open(models_path +modelname , 'rb'))


print 'Preparation sequence COMPLETE. '
print 'The reconstruction will be conducted with  the model ' + modelname 


#Apply the chosen model to the original cropped files in rcstr_path. 
print 'Initiating the reconstruction sequence... ' 
cuda.init()
for j in xrange(1, 101):
    print 'Loading the reconstruction set for the slice %s' %str(j) + '...'
    
    #if(multicrop == True):
    #      filename = "%s_membrane%s%03d.pkl" % (str(resolution), str(cropsize), j)
    #else:
    #    filename = "multi_crop%s_membrane%s%03d.pkl" % (str(resolution), str(cropsize), j)        
    #datafilepath  = ppr.make_rcstr_set(rcstr_path ,filename, file_index = j, patchsize = cropsize)
    # read the pickle file
    #dataset = pickle.load(open(datafilepath))
    
    #dataset = ppr.make_rcstr_set(rcstr_path ,filename, file_index = j, patchsize = cropsize)
    dataset = ppr.make_test_set(rcstr_path , file_index = j, patchsize = cropsize,is_rcstr = True)
    x_test = dataset['x_test']/255.　#255 is the color intensity of the binary black
    y_test = dataset['y_test']
    
    ##### reshape x for cnn  #####
    x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1],x_test.shape[1]))
    

    prediction_vector = np.zeros_like(y_test, dtype = float) #CHOO KAKKEEE

    testsize = x_test.shape[0]
    minibatchsize = 1000
    confusion_matrix = np.zeros((2,2))
    
    #Multiple cropping
    # Reason for copying can be found in http://comments.gmane.org/gmane.comp.python.cuda/3561
    x_test_tr = np.transpose(x_test, (0,1,3,2)).copy()
    x_test_lr  = x_test[:,:,:,::-1].copy()
    x_test_ud  = x_test[:,:,::-1,:].copy()
    x_test_udtr = np.transpose(x_test_ud, (0,1,3,2)).copy()    
    
    for i in xrange(0, testsize, minibatchsize):
        x_batch = cuda.to_gpu(x_test[i : i + minibatchsize])
        y_batch = cuda.to_gpu(y_test[i : i + minibatchsize])
        
        loss, accuracy, prob = model_rc.forward(x_batch, y_batch,train=False)
        if(multicrop == True):
            
            #Data Augmentation
            x_batch_tr = cuda.to_gpu(x_test_tr[i : i + minibatchsize])
            y_batch_tr = cuda.to_gpu(y_test[i : i + minibatchsize])    
            loss_tr, accuracy_tr, prob_tr = model_rc.forward(x_batch_tr, y_batch_tr,train=False)

            x_batch_lr = cuda.to_gpu(x_test_lr[i : i + minibatchsize])
            y_batch_lr = cuda.to_gpu(y_test[i : i + minibatchsize])
            loss_lr, accuracy_lr, prob_lr = model_rc.forward(x_batch_lr, y_batch_lr,train=False)

            x_batch_ud = cuda.to_gpu(x_test_ud[i : i + minibatchsize])
            y_batch_ud = cuda.to_gpu(y_test[i : i + minibatchsize])
            loss_ud, accuracy_ud, prob_ud = model_rc.forward(x_batch_ud, y_batch_ud,train=False)

            x_batch_udtr = cuda.to_gpu(x_test_udtr[i : i + minibatchsize])
            y_batch_udtr = cuda.to_gpu(y_test[i : i + minibatchsize])
            loss_udtr, accuracy_udtr, prob_udtr = model_rc.forward(x_batch_udtr, y_batch_udtr,train=False)

            #MUST be CPUtized before processing the output (GPU format insisits float32 array format) 
            prob_data =   cuda.to_cpu(prob.data)[:,1] 
            prob_tr_data = cuda.to_cpu(prob_tr.data)[:,1]
            prob_lr_data = cuda.to_cpu(prob_lr.data)[:,1]
            prob_ud_data = cuda.to_cpu(prob_ud.data)[:,1]
            prob_udtr_data = cuda.to_cpu(prob_udtr.data)[:,1]        
            prob_average =  np.mean([prob_data, prob_tr_data, prob_lr_data, prob_ud_data, prob_udtr_data], axis = 0) 
        else:
            prob_average = prob_data
        #prediction = cuda.to_cpu(prob_average[:,1] )
        prediction =  prob_average
        #pred = np.argmax(cuda.to_cpu(prob.data),axis=1)
        prediction_vector[i: i + minibatchsize] = prediction
    
        #calc confusion matrix

        print 'batch number', i
        

    prediction_array = prediction_vector.reshape((256-cropsize,256-cropsize)) 
    prediction_array_binary = (prediction_vector.reshape((256-cropsize,256-cropsize))  > threshold)*1.0
    
    prediction_image = Image.fromarray(np.uint8(prediction_array * 255))
    prediction_image_binary = Image.fromarray(np.uint8(prediction_array_binary  * 255))
    
    if(multicrop ==True):
        prediction_image.save(save_path + "multi_crop_prediction_image_%s_%03d.tif" % (str(resolution),j) )
        prediction_image_binary.save(save_path+ "multi_crop_prediction_binary_image_%s_%03d.tif" % (str(resolution),j) )
    else:
        prediction_image.save(save_path + "prediction_image_%s_%03d.tif" % (str(resolution),j) )
        prediction_image_binary.save(save_path+ "prediction_binary_image_%s_%03d.tif" % (str(resolution),j) )        

print 'COMPLETE.  Reconstructed data saved in: \n'
print save_path                                 
                                 



