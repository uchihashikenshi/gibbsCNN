## This makes the dataset ready for training and testing

# train path will be  ...  train_path 
# test path will be   ...  test_path 

def make_dataset(train_path, test_path, save_destination_path, patchsize = 33):
    import numpy as np
    from PIL import Image
    import pickle
    
    imgnames_and_labels_train = np.loadtxt(open(train_path+'training.txt'),dtype=np.str)
    imgnames_and_labels_test = np.loadtxt(open(test_path+'test.txt'),dtype=np.str)
    
    #patchsize = 33
    N_train = imgnames_and_labels_train.shape[0]
    N_test = imgnames_and_labels_test.shape[0]
    print N_train
    print N_test
    x_train = np.zeros((N_train,patchsize,patchsize),dtype='float32')
    x_test = np.zeros((N_test,patchsize,patchsize),dtype='float32')
    y_train = np.zeros((N_train,),'int32')
    y_test = np.zeros((N_test,),'int32')
    
    for i in xrange(0,N_train,1):
        im = Image.open(train_path + imgnames_and_labels_train[i,0])
        x_train[i] = np.array(im)
        y_train[i] = np.int(imgnames_and_labels_train[i,1])
        if(i%5000==0):
            print 'now index :' + str(i)
            
        
    for i in xrange(0,N_test,1):
        im = Image.open(test_path + imgnames_and_labels_test[i,0])
        x_test[i] = np.array(im)
        y_test[i] = np.int(imgnames_and_labels_test[i,1])
        if(i%5000==0):
            print 'now index :' + str(i)
    
    dataset = {'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}
    pickle.dump(dataset,open(save_destination_path+'256_membrane%s.pkl' %str(patchsize),'wb'))
    
