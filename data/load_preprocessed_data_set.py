## This returns the dataset x_test (N_test x patches x patthes)  and y_test (Ntext) 

# train path will be  ...  train_path 
# test path will be   ...  test_path 
#parent_path = "/home/koyama-m/Research/membrane_CNN/data/" 




def make_data_set(data_path,  patchsize = 33,label_data_path = '',  file_index = 0, is_training =False, is_rcstr = False, boring_size = 0):
    
    import numpy as np
    from PIL import Image
    import pickle
    import re
    
    if(is_rcstr):
        imgnames_and_labels_data = np.loadtxt(open(data_path+'test%03d.txt' %file_index),dtype=np.str)
    else:   
        if(is_training):
            imgnames_and_labels_data = np.loadtxt(open(data_path+'training.txt'),dtype=np.str)
        else:
            imgnames_and_labels_data = np.loadtxt(open(data_path+'test.txt'),dtype=np.str)

    
    #patchsize = 33
    center = (patchsize - 1) / 2
    N_data = imgnames_and_labels_data.shape[0]
    print N_data
    x_data = np.zeros((N_data,patchsize,patchsize),dtype='float32')
    label_x_data = np.zeros((N_data,patchsize,patchsize),dtype='float32')    
    y_data = np.zeros((N_data,),'int32')
        
    
    if(len(label_data_path) == 0):
        for i in xrange(0,N_data,1):
            im = Image.open(data_path + imgnames_and_labels_data[i,0])
            imlabel =  Image.open(data_path + 'label_' + imgnames_and_labels_data[i,0])
            x_data[i] = np.array(list(im.getdata())).reshape(patchsize, patchsize)
            y_data[i] = np.int(imgnames_and_labels_data[i,1])
            if(i%5000==0):
                print 'now index :' + str(i)
                
        dataset = {'x_data':x_data,'y_data':y_data}
        return dataset
    else:
        for i in xrange(0,N_data,1):
            im = Image.open(data_path + imgnames_and_labels_data[i,0])
            label_im = Image.open(label_data_path +'label_' + imgnames_and_labels_data[i,0])
            
            x_data[i] = np.array(list(im.getdata())).reshape(patchsize, patchsize)     
            label_x_data[i] = np.array(list(label_im.getdata())).reshape(patchsize, patchsize) 
            
            
            if(boring_size > 0):
                boring_core = np.zeros((2*boring_size+1,2*boring_size+1)) 
                label_x_data[i,center-boring_size:center +boring_size+1,center-boring_size:center +boring_size+1] = boring_core
            else:
                label_x_data[i][center][center] = 0.

            y_data[i] = np.int(imgnames_and_labels_data[i,1])
            
            if(i%5000==0):
                print 'now index :' + str(i)
               
        dataset = {'x_data':x_data,'label_x_data': label_x_data, 'y_data':y_data}
        return dataset    

    
    
    
