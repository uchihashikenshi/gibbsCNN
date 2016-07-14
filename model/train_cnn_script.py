

class cnn(object): 
    def __init__(self, crop = 33,  resol = 256):
        self.crop = crop  
        self.resol = resol
        pass
    
    def train(self, epoch):        
        #################
        ####Prepare Data#### 
        #################
        print '=================================================='
        print '========This file will use the model file convnet_trial.py========='
        print '=================================================='

        cropsize = self.crop
        epoch_size = epoch
        resolution =  self.resol
        
        prefix = '/home/koyama-m/Research/membrane_CNN/'
        train_path  = prefix + 'data/training_dataset/256_training_dataset_crop' + str(cropsize) + '/'
        test_path = prefix + 'data/test_dataset/256_test_dataset_crop' + str(cropsize) + '/'
        models_path = prefix + 'models/' 
        data_path = prefix + 'data/'
        pkldata_path = data_path + 'data_pklformat_for_training/'
        sys.path.append(prefix)
        sys.path.append(train_path)
        sys.path.append(test_path)
        sys.path.append(models_path)
        sys.path.append(data_path)

        #######################
        ###### Setup############
        #######################
        print 'importing the required modules ... '

        import sys,pickle
        import numpy as np
        from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
        import chainer.functions as F
        from matplotlib import pyplot as plt
        import logistic_reg
        import convnet_trial as convnet

        from pickle_preprocessed_dataset import  make_dataset
        import os
        reload(logistic_reg)
        reload(convnet)


        print 'COMPLETE \n' 
        ###### load dataset ######
        print 'Loading the preprocessed test data and training data ... '


        file_path = pkldata_path+'256_membrane%s.pkl' %str(cropsize)
        #If the filepath does not exist, create it. 
        if os.path.exists(file_path) != True:
            print('pkl formatted test data and train data does not exist for these parameters ... creating the pkl file...')
            save_destination = pkldata_path
            train_path = data_path + 'training_dataset/%s_training_dataset_crop%s/'%(str(resolution),str(cropsize) )
            test_path = data_path + 'test_dataset/%s_test_dataset_crop%s/'%(str(resolution),str(cropsize))
            make_dataset(train_path, test_path, save_destination, patchsize = cropsize)
        else:
            print('pkl data found. Loading the pkl data...')

        dataset = pickle.load(open(file_path))
        x_train0 = dataset['x_train']/255.
        y_train0 = dataset['y_train']
        x_test = dataset['x_test']/255.
        y_test = dataset['y_test']

        print 'COMPLETE \n '

        ##### Validation Set and  the rest #### 

        print 'Preparing the Validation data... '

        neg_index = np.where(y_train0 == 0)[0]
        pos_index = np.where(y_train0 == 1)[0]
        neg_pos_prop = [neg_index.shape[0], pos_index.shape[0] ]
        print 'Pos Neg proportion in training data ' + str(neg_pos_prop)

        validate_index = np.arange(150000, y_train0.shape[0],1)
        train_index  = np.arange(0,150000,1)

        x_valid = x_train0[validate_index]
        y_valid = y_train0[validate_index]
        x_train = x_train0[train_index]
        y_train = y_train0[train_index]
        print 'COMPLETE \n '

        ##### Tue if hte model is cnn  #####
        print 'Reshaping the dataset '


        model_is_cnn = True

        ##### reshape x for cnn  #####
        if(model_is_cnn==True):
            x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1],x_train.shape[1]))
            x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1],x_test.shape[1]))
            x_valid = x_valid.reshape((x_valid.shape[0],1,x_valid.shape[1],x_valid.shape[1]))
        ######### init GPU status #######
        cuda.init()

        #FXN MUST BE DEFINED BEFORE INITIALIZATION 
        ######## init models ########
        if(model_is_cnn==True):
            model_cpu_ver = convnet.convnet_trial(patchsize=x_train.shape[2])
            model =  convnet.convnet_trial(patchsize=x_train.shape[2]).to_gpu()
        else:
            model =  logistic_reg.logistic_r(patchsize=x_train.shape[1]).to_gpu()
        print 'COMPLETE \n '

        ######## init optimizer #######
        print 'Initializing the optimizer...\n '

        optimizer = optimizers.Adam()
        optimizer.setup(model.collect_parameters())
        optimizer.zero_grads()

        print 'Initiating the Training Sequence...'

        #######################
        ######Training###########
        #######################

        import time

        trainsize = x_train.shape[0]
        validsize = x_valid.shape[0]

        #Data Augmentation
        x_traintr = np.transpose(x_train, (0,1,3,2))
        x_trainlr  = x_train[:,:,:,::-1]
        x_trainud  = x_train[:,:,::-1,:]
        x_train_udtr = np.transpose(x_trainud, (0,1,3,2))

        start_time = time.time()

        minibatchsize = 50
        for epoch in xrange(epoch_size):

            elapsed_time = time.time() - start_time
            print 'Elapsed time is ' + str(elapsed_time)
            start_time = time.time()


            indexes = np.random.permutation(trainsize)
            n_batch = indexes.shape[0]/minibatchsize
            sum_loss = 0
            sum_accuracy = 0
            for i in xrange(0, trainsize, minibatchsize):

                x_batch_orig = x_train[indexes[i : i + minibatchsize]]
                y_batch_orig = y_train[indexes[i : i + minibatchsize]]

                x_batch_tr = x_traintr[indexes[i : i + minibatchsize]]
                y_batch_tr = y_train[indexes[i : i + minibatchsize]]        

                x_batch_lr = x_trainlr[indexes[i : i + minibatchsize]]
                y_batch_lr = y_train[indexes[i : i + minibatchsize]]

                x_batch_ud = x_trainud[indexes[i : i + minibatchsize]]
                y_batch_ud = y_train[indexes[i : i + minibatchsize]]

                x_batch_udtr = x_train_udtr[indexes[i : i + minibatchsize]]
                y_batch_udtr = y_train[indexes[i : i + minibatchsize]]        

                pre_x_batch = np.concatenate((x_batch_orig,x_batch_tr, x_batch_lr, x_batch_ud, x_batch_udtr), axis=0)
                pre_y_batch = np.concatenate((y_batch_orig,y_batch_tr, y_batch_lr, y_batch_ud,y_batch_udtr), axis=0)        

                x_batch = cuda.to_gpu(pre_x_batch)
                y_batch = cuda.to_gpu(pre_y_batch)
                optimizer.zero_grads()

                loss, accuracy,pred = model.forward(x_batch, y_batch)

                sum_loss += loss.data*minibatchsize
                sum_accuracy += accuracy.data*minibatchsize
                loss.backward()
                optimizer.update()
                #print 'train loss:' + str(loss.data)
                #print 'train accuracy(%)' + str(accuracy.data)

            sum_val_loss = 0
            sum_val_accuracy = 0
            for i in xrange(0,validsize,minibatchsize):
                x_batch = cuda.to_gpu(x_valid[i : i + minibatchsize])
                y_batch = cuda.to_gpu(y_valid[i : i + minibatchsize])
                loss, accuracy,pred = model.forward(x_batch, y_batch,False)
                sum_val_loss += loss.data*minibatchsize
                sum_val_accuracy += accuracy.data*minibatchsize

            print 'epoch ', epoch
            print 'train loss:' + str(sum_loss/trainsize)
            print 'train accuracy(%)' + str(sum_accuracy/trainsize*100)
            print 'validation loss' + str(sum_val_loss/validsize)    
            print 'validation accuracy(%)' + str(sum_val_accuracy/validsize*100)    


            print type(model)
            modelname = 'trained_model%s_crop%sepoch%s.pkl' %(str(resolution),  str(cropsize), str(epoch_size))
            print modelname
            pickle.dump(model, open(models_path+ modelname,'wb'),-1)

        elapsed_time = time.time() - start_time
        print elapsed_time
        print 'Training sequence COMPLETE'  


        print 'Initiating the Testing Sequence...'

        #######################
        ###### Testing ###########
        #######################
        testsize = x_test.shape[0]
        minibatchsize = 1000
        sum_loss = 0
        sum_accuracy = 0
        confusion_matrix = np.zeros((2,2))
        for i in xrange(0, testsize, minibatchsize):
                x_batch = cuda.to_gpu(x_test[i : i + minibatchsize])
                y_batch = cuda.to_gpu(y_test[i : i + minibatchsize])
                loss, accuracy, prob = model.forward(x_batch, y_batch,train=False)
                sum_loss += loss.data*minibatchsize
                sum_accuracy += accuracy.data*x_batch.shape[0]
                #pred = cuda.to_cpu(prob.data)[:,0]>threshold
                pred = np.argmax(cuda.to_cpu(prob.data),axis=1)

                #calc confusion matrix
                for j in xrange(x_batch.shape[0]):
                    confusion_matrix[cuda.to_cpu(y_batch)[j],pred[j]] += 1

        print 'Testing sequence COMPLETE... saving the log... '  
        txtname = 'trained_model%s_crop%sepoch%s_log.txt' %(str(resolution),  str(cropsize), str(epoch_size))
        sys.stdout = open(models_path+ txtname,"w")                        
        print 'test loss:' + str(sum_loss/testsize)

        print 'chance lebel(accuracy)' + str((np.sum(confusion_matrix[0,:])/np.sum(confusion_matrix)))
        print 'test accuracy(%)' + str((confusion_matrix[0,0]+confusion_matrix[1,1])/np.sum(confusion_matrix))

        print 'confusion_matrix:'
        print confusion_matrix

        sys.stdout.close()

    def prepare_data(self, prefix): #I do not know how to share the model across the subroutines without making
        #the x_train to be an attribute so that I can make model to be an attribute as well 
        pass    
    def test_data(self, prefix, x_test):
        pass
    def train_data(self, minibatchsize):       
        pass    
